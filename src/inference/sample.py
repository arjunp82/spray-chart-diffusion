"""Conditional sampling: generate spray charts from batter + situation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.model.diffusion import SprayChartDiffusion
from src.model.unet import ConditionalUNet
from src.data.preprocess import SITUATION_CODES


def load_model(
    checkpoint_path: str | Path,
    processed_dir: str | Path,
    device: torch.device | None = None,
    inpaint_mode: bool = False,
) -> SprayChartDiffusion:
    """Load a trained SprayChartDiffusion model from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    with open(Path(processed_dir) / "batter_id_map.json") as f:
        num_batters = len(json.load(f))

    unet = ConditionalUNet(num_batters=num_batters, inpaint_mode=inpaint_mode).to(device)
    model = SprayChartDiffusion(unet=unet).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate(
    model: SprayChartDiffusion,
    batter_idx: int,
    situation_str: str = "full",   # e.g. "even_R_fastball" or "full"
    num_samples: int = 10,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 200,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Generate `num_samples` spray charts for a given batter and situation.
    Returns (num_samples, 64, 64) numpy array.
    """
    if device is None:
        device = next(model.parameters()).device

    sit_code = SITUATION_CODES.get(situation_str, 12)   # 12 = full-season

    bat_t = torch.tensor([batter_idx] * num_samples, device=device)
    sit_t = torch.tensor([sit_code] * num_samples, device=device)

    with torch.no_grad():
        imgs = model.sample(
            bat_t, sit_t,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
    return imgs.cpu().numpy()[:, 0]   # (N, 64, 64)
