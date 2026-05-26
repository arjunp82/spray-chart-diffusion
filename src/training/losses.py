"""Loss functions for spray chart diffusion training."""

import torch
import torch.nn.functional as F


def epsilon_mse_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """Standard DDPM epsilon-prediction MSE loss."""
    return F.mse_loss(pred_noise, target_noise)


def weighted_epsilon_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-sample weighted MSE — useful for curriculum or importance weighting."""
    loss_per_sample = ((pred_noise - target_noise) ** 2).mean(dim=(1, 2, 3))
    return (loss_per_sample * weights).mean()
