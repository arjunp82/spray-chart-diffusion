"""
Post-training evaluation:
  1. Generate spray charts for 3 batters (mean + uncertainty)
  2. Sparse-data test (KL vs PA count) — diffusion vs HistoricalKDE, multi-batter
  3. Calibration reliability diagram — does the model's 80% credible region contain 80% of real hits?
  4. Inpainting gallery
  5. Save all plots as PNGs

Architecture note (matches unet.py exactly):
  The timestep embedding is injected additively inside every ResBlock via a learned linear
  projection (t_emb → channel bias). Batter and situation embeddings are used only at the
  8×8 bottleneck via two separate cross-attention layers (one for batter, one for situation).
  They do NOT flow through the encoder/decoder ResBlocks. Classifier-free guidance zeroes
  the situation embedding at a 10% rate during training; at inference it runs two forward
  passes (conditioned and unconditional) and interpolates with guidance_scale=5.0.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import normalize_coords, coords_to_image, FAIR_MASK
from src.data.dataset import DENSITY_SCALE
from src.evaluation.metrics import kl_divergence, calibration_score
from src.evaluation.baselines import HistoricalKDE
from src.model.diffusion import SprayChartDiffusion
from src.model.unet import ConditionalUNet
from src.analysis.visualize import plot_spray_chart, overlay_zone_pcts, _FIELD_CMAP


import argparse
import yaml

# ---------------------------------------------------------------------------
# Config (loaded from YAML; defaults to fast config)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", default="configs/default.yaml")
_args, _ = _parser.parse_known_args()

with open(_args.config) as _f:
    _cfg = yaml.safe_load(_f)

PROCESSED_DIR = Path(_cfg["data"]["processed_dir"])
RAW_DIR       = Path(_cfg["data"]["raw_dir"])
CHECKPOINT    = Path(_cfg.get("checkpoint_dir", "checkpoints")) / "best.pt"
OUT_DIR       = Path("results")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {DEVICE}")
print(f"Config: {_args.config}  |  Checkpoint: {CHECKPOINT}")

BATTERS = {
    "Freddie Freeman": 518692,  # LHH, oppo field
    "Javier Baez":     595879,  # RHH, pull hitter
    "Jeff McNeil":     643446,  # LHH, contact/spray
}

# Held-out batters: excluded from ALL training data, use embedding index 0
# (population prior) at inference. Defined in configs/default.yaml; duplicated
# here for clarity. These are the only valid batters for sparse KL and calibration.
HELD_OUT_MLBAMS = _cfg["evaluation"]["held_out_batters"]
POPULATION_PRIOR_IDX = 0   # embedding index for held-out / unknown batters

NUM_SAMPLES    = int(os.environ.get("NUM_SAMPLES", _cfg["evaluation"]["num_samples"]))
INFER_STEPS    = int(os.environ.get("INFER_STEPS", 200))
GUIDANCE_SCALE = 5.0

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model():
    with open(PROCESSED_DIR / "batter_id_map.json") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    num_batters = max(id_map.values()) + 1

    mcfg = _cfg["model"]
    unet = ConditionalUNet(
        num_batters=num_batters,
        base_channels=mcfg["base_channels"],
        channel_multipliers=tuple(mcfg["channel_multipliers"]),
        batter_embed_dim=mcfg["batter_embedding_dim"],
        situation_embed_dim=mcfg["situation_embedding_dim"],
        time_embed_dim=mcfg["timestep_embedding_dim"],
        inpaint_mode=True,
    ).to(DEVICE)

    model = SprayChartDiffusion(unet=unet).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
    return model, id_map


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------
def generate_samples(model, batter_idx: int, situation_code: int = 12,
                     n: int = NUM_SAMPLES) -> np.ndarray:
    bat = torch.tensor([batter_idx] * n, device=DEVICE)
    sit = torch.tensor([situation_code] * n, device=DEVICE)
    with torch.no_grad():
        imgs = model.sample(bat, sit, num_inference_steps=INFER_STEPS,
                            guidance_scale=GUIDANCE_SCALE, device=DEVICE)
    return imgs.cpu().numpy()[:, 0]   # (N, 64, 64)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
CMAP_STD = plt.get_cmap("Blues")

def add_field_wedge(ax):
    """Overlay foul lines and outfield arc on a normalised [-1,1] axes."""
    theta = np.linspace(np.pi * 0.25, np.pi * 0.75, 120)
    ax.plot(np.cos(theta), np.sin(theta), "w-", lw=0.8, alpha=0.6)
    ax.plot([0, -1.0], [0, 1.0], "w-", lw=0.8, alpha=0.6)
    ax.plot([0,  1.0], [0, 1.0], "w-", lw=0.8, alpha=0.6)


def show_density(ax, density, title, cmap=_FIELD_CMAP, vmax=None):
    if vmax is None:
        nonzero = density[density > 0]
        if nonzero.size > 0:
            vmax = float(np.percentile(nonzero, 98))
        else:
            vmax = 1e-6
    ax.imshow(density, extent=[-1, 1, -1, 1], origin="upper",
              cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    add_field_wedge(ax)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)


# ---------------------------------------------------------------------------
# Task 1 — Spray charts for 3 batters
# ---------------------------------------------------------------------------
def task_batter_comparison(model, id_map):
    print("\n=== Task 1: Generating spray charts for 3 batters ===")

    results = {}
    for name, mlbam in BATTERS.items():
        bidx = id_map.get(mlbam)
        if bidx is None:
            print(f"  {name} ({mlbam}) not in dataset — skipping")
            continue
        print(f"  Generating {NUM_SAMPLES} samples for {name} (embed row {bidx})...")
        samples = generate_samples(model, bidx)
        results[name] = {
            "mean": samples.mean(axis=0),
            "std":  samples.std(axis=0),
            "samples": samples,
        }
        print(f"    mean density sum: {results[name]['mean'].sum():.4f}  "
              f"max: {results[name]['mean'].max():.4f}")

    n_batters = len(results)
    fig = plt.figure(figsize=(5 * n_batters, 11))
    gs = gridspec.GridSpec(3, n_batters, figure=fig, hspace=0.35, wspace=0.15)

    for col, (name, d) in enumerate(results.items()):
        ax_mean = fig.add_subplot(gs[0, col])
        show_density(ax_mean, d["mean"], f"{name}\nMean spray chart")
        overlay_zone_pcts(d["mean"], ax_mean)
        show_density(fig.add_subplot(gs[1, col]), d["std"],
                     "Uncertainty (std)", cmap=CMAP_STD)
        show_density(fig.add_subplot(gs[2, col]), d["samples"][0],
                     "Single sample draw")

    fig.suptitle("Spray Chart Diffusion — 3 Batters\n"
                 f"({NUM_SAMPLES} samples each, {INFER_STEPS} denoising steps)",
                 fontsize=11, y=1.01)

    path = OUT_DIR / "batter_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return results


# ---------------------------------------------------------------------------
# Also save individual batter figures
# ---------------------------------------------------------------------------
def task_individual_plots(results):
    for name, d in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        show_density(axes[0], d["mean"], f"{name} — Mean")
        overlay_zone_pcts(d["mean"], axes[0])
        show_density(axes[1], d["std"],  f"{name} — Uncertainty", cmap=CMAP_STD)
        fig.tight_layout()
        safe = name.lower().replace(" ", "_")
        path = OUT_DIR / f"batter_{safe}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Task 2 — Sparse data test: diffusion vs HistoricalKDE, held-out batters
#
# Clean evaluation design:
#   - Batters are from HELD_OUT_MLBAMS — completely excluded from training data
#     and from the embedding table (they use index 0, the population prior).
#   - For each trial, k events are drawn as the conditioning partial chart.
#     Ground truth is the density built from the remaining held-out events.
#   - Both diffusion and KDE see ONLY the k conditioning events.
#   - An embedding-only control (no partial chart, index 0) quantifies how much
#     the k observed PAs add on top of the population prior alone.
# ---------------------------------------------------------------------------
def task_sparse_data(model, id_map):
    print("\n=== Task 2: Sparse data test (held-out batters, population-prior embedding) ===")

    raw = pd.read_csv(RAW_DIR / "statcast_2023.csv", low_memory=False)

    pa_thresholds = [10, 25, 50, 100]
    MIN_TOTAL = 200   # need enough events for a meaningful held-out target
    n_trials  = 3

    diff_kls:  dict[int, list[float]] = {k: [] for k in pa_thresholds}
    kde_kls:   dict[int, list[float]] = {k: [] for k in pa_thresholds}
    emb_kls:   dict[int, list[float]] = {k: [] for k in pa_thresholds}  # embedding-only control

    batters_run  = 0
    first_events = None   # saved for inpainting gallery

    for mlbam in HELD_OUT_MLBAMS:
        events = raw[raw["batter"] == mlbam].dropna(
            subset=["hc_x", "hc_y", "pitch_type", "p_throws", "balls", "strikes"]
        ).reset_index(drop=True)
        if len(events) < MIN_TOTAL:
            print(f"  batter {mlbam}: only {len(events)} events, skipping")
            continue

        hx = events["hc_x"].values
        hy = events["hc_y"].values
        if first_events is None:
            first_events = (mlbam, events)

        print(f"  batter {mlbam} ({len(events)} events, embed=0)...", end=" ", flush=True)

        for k in pa_thresholds:
            for trial in range(n_trials):
                rng = np.random.default_rng(mlbam + trial * 1000)

                cond_idx  = rng.choice(len(events), k, replace=False)
                held_mask = np.ones(len(events), bool)
                held_mask[cond_idx] = False
                cond_hx, cond_hy = hx[cond_idx], hy[cond_idx]
                held_hx, held_hy = hx[held_mask], hy[held_mask]

                xn_h, yn_h = normalize_coords(held_hx, held_hy)
                held_chart = coords_to_image(xn_h, yn_h)

                xn_c, yn_c = normalize_coords(cond_hx, cond_hy)
                partial_np = coords_to_image(xn_c, yn_c)

                # --- Diffusion inpainting (population prior + k observed PAs) ---
                partial_t = (torch.from_numpy(partial_np)
                             .unsqueeze(0).unsqueeze(0)
                             .repeat(8, 1, 1, 1).to(DEVICE) * DENSITY_SCALE)
                bat = torch.full((8,), POPULATION_PRIOR_IDX, dtype=torch.long, device=DEVICE)
                sit = torch.full((8,), 12, dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    gen = model.inpaint_sample(bat, sit, partial_t,
                                               num_inference_steps=50,
                                               guidance_scale=GUIDANCE_SCALE,
                                               device=DEVICE)
                mean_gen = gen.cpu().numpy()[:, 0].mean(axis=0)
                diff_kls[k].append(kl_divergence(mean_gen, held_chart))

                # --- Embedding-only control (population prior, no partial chart) ---
                zeros_t = torch.zeros(8, 1, 64, 64, device=DEVICE)
                with torch.no_grad():
                    gen_emb = model.inpaint_sample(bat, sit, zeros_t,
                                                   num_inference_steps=50,
                                                   guidance_scale=GUIDANCE_SCALE,
                                                   device=DEVICE)
                mean_emb = gen_emb.cpu().numpy()[:, 0].mean(axis=0)
                emb_kls[k].append(kl_divergence(mean_emb, held_chart))

                # --- HistoricalKDE baseline (fitted on k conditioning events only) ---
                # Uses all k events as a plain KDE — no situation bucketing,
                # so it never collapses to uniform at low k.
                cond_df = events.iloc[cond_idx].copy()
                try:
                    kde_sub = HistoricalKDE()
                    kde_sub.fit(cond_df)
                    kde_pred = kde_sub.predict(mlbam, {})
                except Exception:
                    kde_pred = partial_np
                kde_kls[k].append(kl_divergence(kde_pred, held_chart))

        print("done")
        batters_run += 1

    print(f"\n  Ran {batters_run} held-out batters × {n_trials} trials each")
    print(f"\n  {'k':>5}  {'Diffusion':>12}  {'Embed-only':>12}  {'HistKDE':>12}")
    for k in pa_thresholds:
        dm = np.mean(diff_kls[k]) if diff_kls[k] else float("nan")
        em = np.mean(emb_kls[k])  if emb_kls[k]  else float("nan")
        km = np.mean(kde_kls[k])  if kde_kls[k]  else float("nan")
        print(f"  {k:>5}  {dm:>12.4f}  {em:>12.4f}  {km:>12.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for vals, color, marker, ls, label in [
        (diff_kls, "#0066cc", "o", "-",  "Diffusion (population prior + k PAs)"),
        (emb_kls,  "#6600cc", "^", "--", "Embedding-only control (no observed PAs)"),
        (kde_kls,  "#cc6600", "s", "--", "Historical KDE (k PAs only)"),
    ]:
        ks    = sorted(vals.keys())
        means = [np.mean(vals[k]) for k in ks]
        sems  = [np.std(vals[k]) / max(len(vals[k]) ** 0.5, 1) for k in ks]
        ax.errorbar(ks, means, yerr=sems, fmt=f"{marker}{ls}", color=color,
                    lw=2, ms=7, capsize=4, label=label)

    ax.set_xlabel("Conditioning plate appearances (k)", fontsize=11)
    ax.set_ylabel("KL divergence  ↓ better", fontsize=11)
    ax.set_title(
        f"Sparse-Data Stress Test  ({batters_run} held-out batters, {n_trials} trials)\n"
        "Batters unseen at training — embedding index 0 (population prior)",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "sparse_data_kl.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # Return full chart for inpainting gallery
    if first_events is not None:
        _, fe = first_events
        xn, yn = normalize_coords(fe["hc_x"].values, fe["hc_y"].values)
        full_chart = coords_to_image(xn, yn)
    else:
        fe = raw[raw["batter"] == HELD_OUT_MLBAMS[0]].dropna(subset=["hc_x", "hc_y"])
        xn, yn = normalize_coords(fe["hc_x"].values, fe["hc_y"].values)
        full_chart = coords_to_image(xn, yn)

    return diff_kls, kde_kls, full_chart


# ---------------------------------------------------------------------------
# Task 3 — Calibration reliability diagram
# ---------------------------------------------------------------------------
def task_calibration(model, id_map):
    print("\n=== Task 3: Calibration reliability diagram ===")

    raw  = pd.read_csv(RAW_DIR / "statcast_2023.csv", low_memory=False)
    meta = pd.read_csv(PROCESSED_DIR / "metadata.csv")

    confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90]
    pa_thresholds     = [25, 50, 100, 200]
    n_trials          = 3
    n_samples         = 20   # samples per calibration call — kept low for speed

    # coverage[conf_level][k] = list of observed coverages across batters/trials
    coverage: dict[float, dict[int, list[float]]] = {
        c: {k: [] for k in pa_thresholds} for c in confidence_levels
    }

    batters_run = 0
    for mlbam in HELD_OUT_MLBAMS:
        events = raw[raw["batter"] == mlbam].dropna(subset=["hc_x", "hc_y"])
        hx, hy = events["hc_x"].values, events["hc_y"].values
        if len(hx) < max(pa_thresholds) + 50:
            print(f"  batter {mlbam}: only {len(hx)} events, skipping")
            continue

        print(f"  batter {mlbam} ({len(hx)} events, embed=0)...", end=" ", flush=True)

        for k in pa_thresholds:
            for trial in range(n_trials):
                rng = np.random.default_rng(mlbam + trial * 997)

                train_idx = rng.choice(len(hx), k, replace=False)
                mask = np.ones(len(hx), bool)
                mask[train_idx] = False
                test_hx = hx[mask]
                test_hy = hy[mask]
                if len(test_hx) < 10:
                    continue

                xn, yn     = normalize_coords(hx[train_idx], hy[train_idx])
                partial_np = coords_to_image(xn, yn)
                partial_t  = (torch.from_numpy(partial_np)
                              .unsqueeze(0).unsqueeze(0)
                              .repeat(n_samples, 1, 1, 1).to(DEVICE) * DENSITY_SCALE)
                # Held-out batters always use population prior (index 0)
                bat = torch.full((n_samples,), POPULATION_PRIOR_IDX,
                                 dtype=torch.long, device=DEVICE)
                sit = torch.full((n_samples,), 12, dtype=torch.long, device=DEVICE)

                with torch.no_grad():
                    gen = model.inpaint_sample(bat, sit, partial_t,
                                               num_inference_steps=50,
                                               guidance_scale=GUIDANCE_SCALE,
                                               device=DEVICE)
                samples = gen.cpu().numpy()[:, 0]   # (n_samples, 64, 64)

                # Normalize held-out actual coords
                xn_test, yn_test = normalize_coords(test_hx, test_hy)
                actual_coords = np.stack([xn_test, yn_test], axis=1)

                for conf in confidence_levels:
                    obs = calibration_score(samples, actual_coords,
                                            confidence_level=conf)
                    coverage[conf][k].append(obs)

        print("done")
        batters_run += 1

    print(f"\n  Ran calibration on {batters_run} batters")

    # ---- Plot 1: reliability diagram at k=100 (one curve per confidence level) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: reliability diagram — expected vs observed at k=100
    k_ref = 100
    exp_covs, obs_means, obs_sems = [], [], []
    for conf in confidence_levels:
        vals = coverage[conf][k_ref]
        if not vals:
            continue
        exp_covs.append(conf)
        obs_means.append(np.mean(vals))
        obs_sems.append(np.std(vals) / max(len(vals) ** 0.5, 1))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.errorbar(exp_covs, obs_means, yerr=obs_sems,
                fmt="o-", color="#0066cc", lw=2, ms=7, capsize=4,
                label=f"Diffusion (k={k_ref})")
    ax.set_xlabel("Expected coverage", fontsize=11)
    ax.set_ylabel("Observed coverage", fontsize=11)
    ax.set_title(f"Reliability diagram  (k={k_ref} PA,  {batters_run} held-out batters)", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)

    # Right: 80% coverage vs k  (how does calibration change with PA count?)
    conf_ref = 0.80
    ks_plot, means_plot, sems_plot = [], [], []
    for k in pa_thresholds:
        vals = coverage[conf_ref][k]
        if not vals:
            continue
        ks_plot.append(k)
        means_plot.append(np.mean(vals))
        sems_plot.append(np.std(vals) / max(len(vals) ** 0.5, 1))

    ax2 = axes[1]
    ax2.axhline(conf_ref, color="gray", linestyle=":", lw=1.5,
                label=f"Target {conf_ref:.0%}")
    ax2.errorbar(ks_plot, means_plot, yerr=sems_plot,
                 fmt="o-", color="#0066cc", lw=2, ms=7, capsize=4,
                 label="Diffusion inpaint")
    ax2.set_xlabel("Conditioning plate appearances (k)", fontsize=11)
    ax2.set_ylabel(f"{conf_ref:.0%} credible region coverage", fontsize=11)
    ax2.set_title(f"{conf_ref:.0%} coverage vs PA count  ({batters_run} held-out batters)", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(alpha=0.3)

    fig.suptitle("Calibration: does the model know what it doesn't know?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = OUT_DIR / "calibration.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    return coverage


# ---------------------------------------------------------------------------
# Task 4 — Inpainting gallery
# ---------------------------------------------------------------------------
def task_inpaint_gallery(model, id_map, full_chart):
    print("\n=== Task 4: Inpainting gallery ===")
    mlbam = 518692
    bidx  = id_map[mlbam]

    raw = pd.read_csv(RAW_DIR / "statcast_2023.csv", low_memory=False)
    events = raw[raw["batter"] == mlbam].dropna(subset=["hc_x", "hc_y"])
    hx, hy = events["hc_x"].values, events["hc_y"].values

    ks = [10, 25, 50, 100]
    fig, axes = plt.subplots(2, len(ks) + 1, figsize=(4 * (len(ks) + 1), 8))

    # Full chart in first column
    show_density(axes[0, 0], full_chart, "Full season\n(ground truth)")
    axes[1, 0].axis("off")

    for col, k in enumerate(ks, start=1):
        if k > len(hx):
            axes[0, col].axis("off"); axes[1, col].axis("off"); continue

        rng = np.random.default_rng(0)
        idx = rng.choice(len(hx), k, replace=False)
        sub_hx, sub_hy = hx[idx], hy[idx]

        xn, yn = normalize_coords(sub_hx, sub_hy)
        partial_np = coords_to_image(xn, yn)

        partial_t = (torch.from_numpy(partial_np)
                     .unsqueeze(0).unsqueeze(0).repeat(10, 1, 1, 1).to(DEVICE) * DENSITY_SCALE)
        bat = torch.tensor([bidx] * 10, device=DEVICE)
        sit = torch.tensor([12]   * 10, device=DEVICE)

        with torch.no_grad():
            gen = model.inpaint_sample(bat, sit, partial_t,
                                       num_inference_steps=100,
                                       guidance_scale=GUIDANCE_SCALE,
                                       device=DEVICE)
        mean_gen = gen.cpu().numpy()[:, 0].mean(axis=0)

        show_density(axes[0, col], partial_np,  f"k={k} observed\n(partial)")
        show_density(axes[1, col], mean_gen, f"Inpainted\n(k={k})")

    fig.suptitle("Freddie Freeman — Inpainting at Different PA Counts", fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "inpaint_gallery.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, id_map = load_model()

    batter_results = task_batter_comparison(model, id_map)
    task_individual_plots(batter_results)

    diff_kls, kde_kls, full_chart = task_sparse_data(model, id_map)
    task_calibration(model, id_map)
    task_inpaint_gallery(model, id_map, full_chart)

    print("\n=== Summary ===")
    print(f"All plots saved to: {OUT_DIR.resolve()}")
    print("\nKL divergence summary (lower is better):")
    print(f"{'k':>6}  {'Diffusion':>12}  {'HistKDE':>10}")
    for k in sorted(diff_kls):
        dm = np.mean(diff_kls[k]) if diff_kls[k] else float("nan")
        km = np.mean(kde_kls[k])  if kde_kls[k]  else float("nan")
        print(f"{k:>6}  {dm:>12.4f}  {km:>10.4f}")
