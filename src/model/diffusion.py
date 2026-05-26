"""DDPM training and sampling logic with classifier-free guidance."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from tqdm import tqdm

from src.model.unet import ConditionalUNet
from src.data.preprocess import FAIR_MASK as _FAIR_MASK_NP

# Precompute FAIR_MASK as a float tensor (1, 1, 64, 64) for broadcasting.
# Re-used across all calls; moved to device in sample/inpaint_sample.
_FAIR_MASK_CPU: torch.Tensor = torch.from_numpy(_FAIR_MASK_NP.astype(np.float32)).unsqueeze(0).unsqueeze(0)


class SprayChartDiffusion(nn.Module):
    """
    Wraps ConditionalUNet with DDPM noise scheduling.

    Training : call loss() → scalar
    Inference: call sample() or inpaint_sample()
    """

    def __init__(
        self,
        unet: ConditionalUNet,
        num_timesteps: int = 1000,
        noise_schedule: str = "squaredcos_cap_v2",   # HF name for cosine
        cfg_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.cfg_dropout_prob = cfg_dropout_prob
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule=noise_schedule,
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noise and add it to x_0 according to the DDPM schedule."""
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, noise, t)
        return x_t, noise

    def loss(
        self,
        x_0: torch.Tensor,
        batter_idx: torch.Tensor,
        situation_code: torch.Tensor,
        partial_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Standard DDPM epsilon-prediction MSE loss with classifier-free guidance.

        Returns a scalar loss tensor.
        """
        B = x_0.size(0)
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device)

        x_t, noise = self.add_noise(x_0, t)

        # Randomly mask situation conditioning for CFG
        cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_prob

        pred_noise = self.unet(
            x_t=x_t,
            timestep=t,
            batter_idx=batter_idx,
            situation_code=situation_code,
            cfg_mask=cfg_mask,
            partial_image=partial_image,
        )

        return nn.functional.mse_loss(pred_noise, noise)

    # ------------------------------------------------------------------
    # Inference — conditional sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batter_idx: torch.Tensor,
        situation_code: torch.Tensor,
        num_inference_steps: int = 1000,
        guidance_scale: float = 3.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion.  Returns (B, 1, 64, 64) spray chart images.
        Uses classifier-free guidance if guidance_scale > 1.
        """
        if device is None:
            device = next(self.unet.parameters()).device

        B = batter_idx.size(0)
        x = torch.randn(B, 1, 64, 64, device=device)

        self.scheduler.set_timesteps(num_inference_steps)

        cfg = guidance_scale > 1.0
        # Full-season unconditional code
        uncond_code = torch.full_like(situation_code, 12)   # 12 = FULL_SEASON_CODE

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            t_batch = t.expand(B).to(device)

            eps_cond = self.unet(x, t_batch, batter_idx, situation_code, cfg_mask=None)

            if cfg:
                eps_uncond = self.unet(
                    x, t_batch, batter_idx, uncond_code,
                    cfg_mask=torch.ones(B, dtype=torch.bool, device=device),
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            x = self.scheduler.step(eps, t, x).prev_sample

        # Clamp and normalise to a valid probability density
        fair = _FAIR_MASK_CPU.to(device)
        x = x.clamp(0, None) * fair
        total = x.sum(dim=(-2, -1), keepdim=True)
        x = x / (total + 1e-8)
        return x

    # ------------------------------------------------------------------
    # Inference — inpainting (sparse data conditioning)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint_sample(
        self,
        batter_idx: torch.Tensor,
        situation_code: torch.Tensor,
        partial_image: torch.Tensor,
        num_inference_steps: int = 1000,
        guidance_scale: float = 3.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion conditioned on a partial spray chart.
        partial_image: (B, 1, 64, 64) density built from k observed events.
        """
        if device is None:
            device = next(self.unet.parameters()).device

        B = batter_idx.size(0)
        x = torch.randn(B, 1, 64, 64, device=device)

        self.scheduler.set_timesteps(num_inference_steps)
        uncond_code = torch.full_like(situation_code, 12)
        cfg = guidance_scale > 1.0

        for t in tqdm(self.scheduler.timesteps, desc="Inpainting", leave=False):
            t_batch = t.expand(B).to(device)

            eps_cond = self.unet(
                x, t_batch, batter_idx, situation_code,
                cfg_mask=None, partial_image=partial_image,
            )

            if cfg:
                eps_uncond = self.unet(
                    x, t_batch, batter_idx, uncond_code,
                    cfg_mask=torch.ones(B, dtype=torch.bool, device=device),
                    partial_image=partial_image,
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            x = self.scheduler.step(eps, t, x).prev_sample

        fair = _FAIR_MASK_CPU.to(device)
        x = x.clamp(0, None) * fair
        total = x.sum(dim=(-2, -1), keepdim=True)
        x = x / (total + 1e-8)
        return x
