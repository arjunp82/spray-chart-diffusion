"""
Generate MLB-style zone charts (wedge regions, solid colors) for each batter
across all 12 situations. Saves one 3x4 grid per batter.
"""

import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Polygon
import matplotlib.colors as mcolors

from src.model.diffusion import SprayChartDiffusion
from src.model.unet import ConditionalUNet
from src.analysis.visualize import MLB_ZONE_MASKS

# ---- Config ---------------------------------------------------------------
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

with open("data/processed/batter_id_map.json") as f:
    id_map = {int(k): v for k, v in json.load(f).items()}

mcfg = cfg["model"]
unet = ConditionalUNet(
    num_batters=len(id_map),
    base_channels=mcfg["base_channels"],
    channel_multipliers=tuple(mcfg["channel_multipliers"]),
    batter_embed_dim=mcfg["batter_embedding_dim"],
    situation_embed_dim=mcfg["situation_embedding_dim"],
    time_embed_dim=mcfg["timestep_embedding_dim"],
    inpaint_mode=True,
).to(DEVICE)
model = SprayChartDiffusion(unet=unet).to(DEVICE)
ckpt = torch.load("checkpoints/default/best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

BATTERS = {
    "Freddie Freeman": id_map[518692],
    "Javier Baez":     id_map[595879],
    "Jeff McNeil":     id_map[643446],
}

SITUATIONS = [
    (0,  "Ahead\nvs LHP FB"),  (1,  "Ahead\nvs LHP OS"),
    (2,  "Ahead\nvs RHP FB"),  (3,  "Ahead\nvs RHP OS"),
    (4,  "Even\nvs LHP FB"),   (5,  "Even\nvs LHP OS"),
    (6,  "Even\nvs RHP FB"),   (7,  "Even\nvs RHP OS"),
    (8,  "Behind\nvs LHP FB"), (9,  "Behind\nvs LHP OS"),
    (10, "Behind\nvs RHP FB"), (11, "Behind\nvs RHP OS"),
]

ZONES = ["LF", "LC", "CF", "RC", "RF", "INF_3B", "INF_LC", "INF_RC", "INF_1B", "SHORT"]

# ---- Drawing function ------------------------------------------------------
def draw_zone_chart(ax, pcts, title="", vmin=0, vmax=40):
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.08, 1.12)
    ax.axis("off")
    ax.set_facecolor("#f0f0f0")

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def c(pct):
        return cmap(0.15 + norm(pct) * 0.75)

    R_OUT   = 1.0
    R_INF   = 0.44
    R_SHORT = 0.18

    # 5 outfield wedges — equal 18° each spanning 45°–135°
    of_bounds = [45, 63, 81, 99, 117, 135]
    of_zones  = ["RF", "RC", "CF", "LC", "LF"]

    for i, zone in enumerate(of_zones):
        t1, t2 = of_bounds[i], of_bounds[i + 1]
        w = Wedge((0, 0), R_OUT, t1, t2, width=R_OUT - R_INF,
                  facecolor=c(pcts[zone]), edgecolor="white", linewidth=2.5, zorder=2)
        ax.add_patch(w)
        mid = np.radians((t1 + t2) / 2)
        r   = (R_INF + R_OUT) / 2
        ax.text(r * np.cos(mid), r * np.sin(mid),
                f"{pcts[zone]:.1f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=3)

    # 4 infield wedges — equal 22.5° each spanning 45°–135°
    inf_bounds = [45, 67.5, 90, 112.5, 135]
    inf_zones  = ["INF_1B", "INF_RC", "INF_LC", "INF_3B"]

    for i, zone in enumerate(inf_zones):
        t1, t2 = inf_bounds[i], inf_bounds[i + 1]
        w = Wedge((0, 0), R_INF, t1, t2, width=R_INF - R_SHORT,
                  facecolor=c(pcts[zone]), edgecolor="white", linewidth=2.5, zorder=2)
        ax.add_patch(w)
        mid = np.radians((t1 + t2) / 2)
        r   = (R_SHORT + R_INF) / 2
        ax.text(r * np.cos(mid), r * np.sin(mid),
                f"{pcts[zone]:.1f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    # Short zone
    w_short = Wedge((0, 0), R_SHORT, 45, 135,
                    facecolor=c(pcts["SHORT"]), edgecolor="white", linewidth=2.5, zorder=2)
    ax.add_patch(w_short)
    ax.text(0, R_SHORT * 0.5,
            f"{pcts['SHORT']:.1f}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=3)

    # Diamond
    d = 0.11
    diamond = Polygon([(0, 0.01), (d, d + 0.01), (0, 2 * d + 0.01), (-d, d + 0.01)],
                      closed=True, fill=False, edgecolor="#555555", linewidth=1.2, zorder=4)
    ax.add_patch(diamond)
    # pitcher's mound
    circ = plt.Circle((0, d + 0.01), 0.025, color="#888888", zorder=4)
    ax.add_patch(circ)

    ax.set_title(title, fontsize=9, pad=4, fontweight="bold")


# ---- Generate + plot -------------------------------------------------------
Path("results").mkdir(exist_ok=True)

for name, bidx in BATTERS.items():
    print(f"\nGenerating {name}...")

    # collect zone pcts for all 12 situations
    all_pcts = {}
    for code, label in SITUATIONS:
        bat = torch.tensor([bidx] * 20, device=DEVICE)
        sit = torch.tensor([code]  * 20, device=DEVICE)
        with torch.no_grad():
            imgs = model.sample(bat, sit, num_inference_steps=50,
                                guidance_scale=5.0, device=DEVICE)
        mean = imgs.cpu().numpy()[:, 0].mean(0)
        all_pcts[code] = {z: float(mean[mask].sum() * 100)
                          for z, mask in MLB_ZONE_MASKS.items()}
        print(f"  sit {code:2d} done")

    # shared scale across all situations for this batter
    all_vals = [v for p in all_pcts.values() for v in p.values()]
    vmax = max(all_vals) * 1.05

    fig, axes = plt.subplots(3, 4, figsize=(20, 16),
                             facecolor="white")
    COUNT_LABELS = ["Ahead in Count", "Even Count", "Behind in Count"]
    PITCH_LABELS = ["vs LHP Fastball", "vs LHP Offspeed",
                    "vs RHP Fastball", "vs RHP Offspeed"]

    for ci in range(3):
        for pi in range(4):
            code = ci * 4 + pi
            ax = axes[ci, pi]
            title = PITCH_LABELS[pi] if ci == 0 else ""
            draw_zone_chart(ax, all_pcts[code], title=title, vmin=0, vmax=vmax)
            if pi == 0:
                ax.text(-0.12, 0.5, COUNT_LABELS[ci],
                        transform=ax.transAxes, rotation=90,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold")

    fig.suptitle(f"{name} — Zone % by Situation  (20 samples · guidance=5.0)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)

    safe = name.lower().replace(" ", "_")
    out = f"results/zone_style_{safe}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

print("\nAll done.")
