"""
Generate a grid of spray charts for one batter across all 12 situations.
Usage:
    python3 scripts/situation_grid.py --batter "Mike Trout"
    python3 scripts/situation_grid.py --batter "Freddie Freeman"
    python3 scripts/situation_grid.py --batter "Steven Kwan"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.diffusion import SprayChartDiffusion
from src.model.unet import ConditionalUNet

# ---- Config ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config",  default="configs/default.yaml")
parser.add_argument("--batter",  default="Mike Trout")
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--steps",   type=int, default=50)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
PROCESSED_DIR = Path(cfg["data"]["processed_dir"])
CKPT          = Path(cfg.get("checkpoint_dir", "checkpoints")) / "best.pt"
OUT_DIR       = Path("results"); OUT_DIR.mkdir(exist_ok=True)

BATTER_MLBAM = {
    "Freddie Freeman": 518692,  # LHH, oppo field
    "Javier Baez":     595879,  # RHH, pull hitter
    "Jeff McNeil":     643446,  # LHH, contact/spray
}

SITUATIONS = [
    (0,  "Ahead vs LHP\nFastball"),
    (1,  "Ahead vs LHP\nOffspeed"),
    (2,  "Ahead vs RHP\nFastball"),
    (3,  "Ahead vs RHP\nOffspeed"),
    (4,  "Even vs LHP\nFastball"),
    (5,  "Even vs LHP\nOffspeed"),
    (6,  "Even vs RHP\nFastball"),
    (7,  "Even vs RHP\nOffspeed"),
    (8,  "Behind vs LHP\nFastball"),
    (9,  "Behind vs LHP\nOffspeed"),
    (10, "Behind vs RHP\nFastball"),
    (11, "Behind vs RHP\nOffspeed"),
    (12, "Full Season"),
]

# ---- Load model -----------------------------------------------------------
with open(PROCESSED_DIR / "batter_id_map.json") as f:
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
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

# ---- Generate -------------------------------------------------------------
mlbam = BATTER_MLBAM.get(args.batter)
if mlbam is None:
    print(f"Unknown batter '{args.batter}'. Choose from: {list(BATTER_MLBAM)}")
    sys.exit(1)
bidx = id_map[mlbam]
print(f"Generating {args.batter} (embed row {bidx}) across {len(SITUATIONS)} situations...")

means = {}
for code, label in SITUATIONS:
    bat = torch.tensor([bidx] * args.samples, device=DEVICE)
    sit = torch.tensor([code]  * args.samples, device=DEVICE)
    with torch.no_grad():
        imgs = model.sample(bat, sit, num_inference_steps=args.steps,
                            guidance_scale=5.0, device=DEVICE)
    means[code] = imgs.cpu().numpy()[:, 0].mean(axis=0)
    print(f"  sit {code:2d} ({label.replace(chr(10),' ')}) done")

# ---- Plot -----------------------------------------------------------------
# Layout: 3 rows of 4 (count state) + 1 full-season panel
# Row labels = count state, col labels = pitcher hand + pitch type
COUNT_LABELS = ["Ahead in Count", "Even Count", "Behind in Count"]
COL_LABELS   = ["vs LHP  Fastball", "vs LHP  Offspeed", "vs RHP  Fastball", "vs RHP  Offspeed"]

ncols = 4
nrows = 4   # 3 count states + 1 full-season row

fig = plt.figure(figsize=(ncols * 4.5, nrows * 4.2))

# Shared vmax across situational panels only (not full-season) for fair comparison
sit_vals = np.concatenate([means[c][means[c] > 0] for c in range(12)])
vmax = float(np.percentile(sit_vals, 98)) if sit_vals.size > 0 else 1e-6

def add_wedge(ax, lw=1.2):
    theta = np.linspace(np.pi * 0.25, np.pi * 0.75, 120)
    ax.plot(np.cos(theta), np.sin(theta), "w-", lw=lw, alpha=0.7)
    ax.plot([0, -1.0], [0, 1.0], "w-", lw=lw, alpha=0.7)
    ax.plot([0,  1.0], [0, 1.0], "w-", lw=lw, alpha=0.7)

def draw_panel(ax, density, title, this_vmax):
    ax.imshow(density, extent=[-1,1,-1,1], origin="upper",
              cmap="hot", vmin=0, vmax=this_vmax, aspect="equal")
    add_wedge(ax)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=5, fontweight="bold")
    # INF % in corner
    from src.data.preprocess import FAIR_MASK
    inf_mask = np.zeros((64,64), bool)
    inf_mask[38:, 22:42] = True
    inf_mask &= FAIR_MASK
    pct = float(density[inf_mask].sum() * 100)
    ax.text(0.02, 0.04, f"INF {pct:.0f}%", transform=ax.transAxes,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4, lw=0))

# Situational panels (3 count states × 4 cols)
gs = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.08,
                      left=0.04, right=0.96, top=0.90, bottom=0.06)

for count_idx, count_label in enumerate(COUNT_LABELS):
    for col_idx, col_label in enumerate(COL_LABELS):
        code = count_idx * 4 + col_idx
        ax = fig.add_subplot(gs[count_idx, col_idx])
        title = col_label if count_idx == 0 else ""
        draw_panel(ax, means[code], title, vmax)
        if col_idx == 0:
            ax.set_ylabel(count_label, fontsize=11, fontweight="bold", labelpad=6)

# Full-season panel — centered in last row
ax_full = fig.add_subplot(gs[3, 1:3])
fs_vals = means[12][means[12] > 0]
fs_vmax = float(np.percentile(fs_vals, 98)) if fs_vals.size > 0 else 1e-6
draw_panel(ax_full, means[12], "Full Season (all situations)", fs_vmax)
fig.add_subplot(gs[3, 0]).axis("off")
fig.add_subplot(gs[3, 3]).axis("off")

fig.suptitle(f"{args.batter} — How Spray Chart Changes by Situation\n"
             f"(guidance scale=5.0 · {args.samples} samples · {args.steps} denoising steps)",
             fontsize=13, y=0.97)

safe = args.batter.lower().replace(" ", "_")
out = OUT_DIR / f"situation_grid_{safe}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")
