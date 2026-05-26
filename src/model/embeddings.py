"""Embedding modules: batter identity, situation, and diffusion timestep."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.data.preprocess import COUNT_STATES, HANDEDNESS, PITCH_GROUPS


class BatterEmbedding(nn.Module):
    """Learned embedding table: one vector per batter (indexed 0-N)."""

    def __init__(self, num_batters: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_batters, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, batter_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(batter_idx)          # (B, embed_dim)


class SituationEmbedding(nn.Module):
    """
    Embeds count_state × pitcher_handedness × pitch_type_group.

    Accepts three integer tensors and concatenates their embeddings before
    projecting to `embed_dim`.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        dim_each = embed_dim // 4              # small per-factor dim before projection
        self.count_emb = nn.Embedding(len(COUNT_STATES), dim_each)
        self.hand_emb = nn.Embedding(len(HANDEDNESS), dim_each)
        self.pitch_emb = nn.Embedding(len(PITCH_GROUPS), dim_each)
        self.proj = nn.Linear(3 * dim_each, embed_dim)

    def forward(
        self,
        count_state: torch.Tensor,     # (B,)  int in {0,1,2}
        pitcher_hand: torch.Tensor,    # (B,)  int in {0,1}
        pitch_type: torch.Tensor,      # (B,)  int in {0,1}
    ) -> torch.Tensor:
        e = torch.cat(
            [self.count_emb(count_state),
             self.hand_emb(pitcher_hand),
             self.pitch_emb(pitch_type)],
            dim=-1,
        )
        return self.proj(e)            # (B, embed_dim)


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep encoding → 2-layer MLP → embed_dim.
    Follows the DDPM / Imagen convention.
    """

    def __init__(self, embed_dim: int = 128, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timestep → (B, freq_dim) sinusoidal features."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, freq_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))    # (B, embed_dim)
