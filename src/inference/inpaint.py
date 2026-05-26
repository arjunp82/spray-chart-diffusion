"""Inpainting inference: recover a full spray chart from k observed batted balls."""

from __future__ import annotations

import numpy as np
import torch

from src.data.preprocess import normalize_coords, coords_to_image
from src.model.diffusion import SprayChartDiffusion
from src.data.preprocess import SITUATION_CODES


def build_partial_chart(
    hc_x: np.ndarray,
    hc_y: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """
    Build a 64×64 partial spray chart from raw Statcast coordinates.
    If k is given, randomly sample k events first.
    """
    if k is not None and k < len(hc_x):
        idx = np.random.choice(len(hc_x), k, replace=False)
        hc_x, hc_y = hc_x[idx], hc_y[idx]
    x, y = normalize_coords(hc_x, hc_y)
    return coords_to_image(x, y)   # (64, 64)


def inpaint(
    model: SprayChartDiffusion,
    batter_idx: int,
    hc_x: np.ndarray,
    hc_y: np.ndarray,
    k: int,
    situation_str: str = "full",
    num_samples: int = 10,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 200,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run inpainting for a batter given k observed batted ball coords.
    Returns (num_samples, 64, 64) array of generated charts.
    """
    if device is None:
        device = next(model.parameters()).device

    partial_np = build_partial_chart(hc_x, hc_y, k=k)
    partial_t = (
        torch.from_numpy(partial_np)
        .unsqueeze(0).unsqueeze(0)
        .repeat(num_samples, 1, 1, 1)
        .to(device)
    )

    sit_code = SITUATION_CODES.get(situation_str, 12)
    bat_t = torch.tensor([batter_idx] * num_samples, device=device)
    sit_t = torch.tensor([sit_code] * num_samples, device=device)

    with torch.no_grad():
        imgs = model.inpaint_sample(
            bat_t, sit_t, partial_t,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
    return imgs.cpu().numpy()[:, 0]   # (N, 64, 64)
