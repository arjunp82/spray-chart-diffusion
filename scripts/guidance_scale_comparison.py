"""
Generate guidance scale comparison grid (3 batters × 4 guidance scales).
Usage:
    python3 scripts/guidance_scale_comparison.py
"""

import json
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

CONFIG  = "configs/default.yaml"
SAMPLES = 10
STEPS   = 50
OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
PROCESSED_DIR = Path(cfg["data"]["processed_dir"])
CKPT          = Path(cfg.get("checkpoint_dir", "checkpoints")) / "best.pt"

BATTERS = {
    "Mike Trout":      545361,
    "Freddie Freeman": 518692,
    "Steven Kwan":     680757,
}
GUIDANCE_SCALES = [1.5, 3.0, 5.0, 8.0]
SITUATION_CODE  = 12   # full-season

with open(PROCESSED_DIR / "batter_id_map.json") as f:
    id_map = {int(k): v for k, v in json.load(f).items()}

mcfg = cfg["model"]
num_batters = max(id_map.values()) + 1
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
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

# Generate all panels
means = {}
for name, mlbam in BATTERS.items():
    bidx = id_map.get(mlbam, 0)
    for gs in GUIDANCE_SCALES:
        print(f"  {name}  gs={gs}")
        bat = torch.tensor([bidx] * SAMPLES, device=DEVICE)
        sit = torch.tensor([SITUATION_CODE] * SAMPLES, device=DEVICE)
        with torch.no_grad():
            imgs = model.sample(bat, sit, num_inference_steps=STEPS,
                                guidance_scale=gs, device=DEVICE)
        means[(name, gs)] = imgs.cpu().numpy()[:, 0].mean(axis=0)

# Plot
nrows = len(BATTERS)
ncols = len(GUIDANCE_SCALES)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.8))

for r, name in enumerate(BATTERS):
    row_vals = np.concatenate([means[(name, gs)][means[(name, gs)] > 0]
                                for gs in GUIDANCE_SCALES])
    vmax = float(np.percentile(row_vals, 98)) if row_vals.size > 0 else 1e-6

    for c, gs in enumerate(GUIDANCE_SCALES):
        ax = axes[r, c]
        ax.imshow(means[(name, gs)], extent=[-1,1,-1,1], origin="upper",
                  cmap="hot", vmin=0, vmax=vmax, aspect="equal")
        # foul lines
        theta = np.linspace(np.pi * 0.25, np.pi * 0.75, 120)
        ax.plot(np.cos(theta), np.sin(theta), "w-", lw=1.0, alpha=0.7)
        ax.plot([0, -1.0], [0, 1.0], "w-", lw=1.0, alpha=0.7)
        ax.plot([0,  1.0], [0, 1.0], "w-", lw=1.0, alpha=0.7)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"gs={gs}", fontsize=11, fontweight="bold")
        if c == 0:
            ax.set_ylabel(name, fontsize=11, fontweight="bold")

fig.suptitle("Guidance Scale Comparison — Same Model, No Retraining", fontsize=13, y=1.01)
fig.tight_layout()

out = OUT_DIR / "guidance_scale_comparison.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")
print(f"Saved → {out.with_suffix('.pdf')}")
