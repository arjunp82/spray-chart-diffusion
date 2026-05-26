"""Conditional 2D U-Net denoiser for spray chart diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import BatterEmbedding, SituationEmbedding, TimestepEmbedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Conv residual block with GroupNorm + SiLU.
    Timestep embedding is injected via additive bias after the first norm.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.dropout(self.conv2(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions (H×W flattened)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class CrossAttention(nn.Module):
    """
    Cross-attention where queries come from the spatial feature map and
    keys/values come from a set of conditioning tokens.
    """

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(cond_dim, 2 * channels)
        self.out_proj = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)   cond: (B, S, cond_dim)
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)

        q = self.q_proj(h)                                     # (B, HW, C)
        kv = self.kv_proj(cond)                                # (B, S, 2C)
        k, v = kv.chunk(2, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, S_, _ = t.shape
            return t.view(B_, S_, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        return x + out.permute(0, 2, 1).view(B, C, H, W)


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(in_ch, out_ch, time_dim, dropout),
            ResBlock(out_ch, out_ch, time_dim, dropout),
        ])
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x, t)
        return self.down(x), x   # downsampled, skip


class DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.blocks = nn.ModuleList([
            ResBlock(in_ch + skip_ch, out_ch, time_dim, dropout),
            ResBlock(out_ch, out_ch, time_dim, dropout),
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for blk in self.blocks:
            x = blk(x, t)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net denoiser.

    Input  x_t : (B, in_channels, 64, 64)  where in_channels=1 normally,
                  or 2 when a partial_image is concatenated for inpainting.
    Output ε̂   : (B, 1, 64, 64)

    Conditioning:
        batter_idx    : (B,)  — batter identity embedding
        situation_code: (B,)  — encoded situation (unused when cfg_mask=True)
        timestep      : (B,)  — diffusion timestep
        cfg_mask      : (B,)  bool — when True for a sample, zero situation emb
    """

    def __init__(
        self,
        num_batters: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        batter_embed_dim: int = 128,
        situation_embed_dim: int = 128,
        time_embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        inpaint_mode: bool = False,
    ) -> None:
        super().__init__()
        self.inpaint_mode = inpaint_mode
        in_ch = 2 if inpaint_mode else 1

        chs = [base_channels * m for m in channel_multipliers]  # [64,128,256,512]

        # Embedding modules
        self.time_emb = TimestepEmbedding(time_embed_dim)
        self.batter_emb = BatterEmbedding(num_batters, batter_embed_dim)
        self.situation_emb = SituationEmbedding(situation_embed_dim)

        cond_dim = batter_embed_dim + situation_embed_dim   # 256

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # Encoder
        self.enc_levels = nn.ModuleList([
            EncoderLevel(chs[i] if i > 0 else chs[0], chs[i], time_embed_dim, dropout)
            for i in range(len(chs))
        ])
        # Fix: first encoder level input is chs[0] (output of in_conv)
        self.enc_levels[0] = EncoderLevel(chs[0], chs[0], time_embed_dim, dropout)
        for i in range(1, len(chs)):
            self.enc_levels[i] = EncoderLevel(chs[i - 1], chs[i], time_embed_dim, dropout)

        # Bottleneck
        bot_ch = chs[-1]
        self.bot_res1 = ResBlock(bot_ch, bot_ch, time_embed_dim, dropout)
        self.bot_self_attn = SpatialSelfAttention(bot_ch, num_heads)
        self.bot_cross_batter = CrossAttention(bot_ch, batter_embed_dim, num_heads)
        self.bot_cross_sit = CrossAttention(bot_ch, situation_embed_dim, num_heads)
        self.bot_res2 = ResBlock(bot_ch, bot_ch, time_embed_dim, dropout)

        # Decoder — 4 encoder levels → 4 decoder levels.
        # Skips consumed in reverse: enc3(chs[3]), enc2(chs[2]), enc1(chs[1]), enc0(chs[0]).
        self.dec_levels = nn.ModuleList()
        rev = list(reversed(chs))   # [512, 256, 128, 64] for base=64
        for i in range(len(chs)):
            in_ch   = rev[i]
            skip_ch = rev[i]
            out_ch  = rev[i + 1] if i + 1 < len(chs) else rev[-1]
            self.dec_levels.append(DecoderLevel(in_ch, skip_ch, out_ch, time_embed_dim, dropout))

        # Output projection
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], 1, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        batter_idx: torch.Tensor,
        situation_code: torch.Tensor,
        cfg_mask: torch.Tensor | None = None,
        partial_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Concatenate partial image for inpainting (zeros if none provided)
        if self.inpaint_mode:
            if partial_image is None:
                partial_image = torch.zeros_like(x_t)
            x_t = torch.cat([x_t, partial_image], dim=1)

        # Embeddings
        t_emb = self.time_emb(timestep)                          # (B, T)

        # Decode situation_code into components for SituationEmbedding
        count_st, hand, pitch_t = _decode_situation_batch(situation_code, x_t.device)
        sit_emb = self.situation_emb(count_st, hand, pitch_t)   # (B, S)

        # Full-season code (12) is always situation-unconditional.
        # CFG mask also zeros situation when explicitly set.
        full_season = (situation_code >= 12)
        if cfg_mask is not None:
            zero_sit = cfg_mask | full_season
        else:
            zero_sit = full_season
        sit_emb = sit_emb * (~zero_sit).float().unsqueeze(-1)

        bat_emb = self.batter_emb(batter_idx)                    # (B, D)

        # Expand conditioning to token sequences for cross-attention
        bat_token = bat_emb.unsqueeze(1)    # (B, 1, D)
        sit_token = sit_emb.unsqueeze(1)    # (B, 1, D)

        # Encoder
        x = self.in_conv(x_t)
        skips = []
        for enc in self.enc_levels:
            x, skip = enc(x, t_emb)
            skips.append(skip)

        # Bottleneck
        x = self.bot_res1(x, t_emb)
        x = self.bot_self_attn(x)
        x = self.bot_cross_batter(x, bat_token)
        x = self.bot_cross_sit(x, sit_token)
        x = self.bot_res2(x, t_emb)

        # Decoder
        for dec, skip in zip(self.dec_levels, reversed(skips)):
            x = dec(x, skip, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ---------------------------------------------------------------------------
# Situation decoding helper
# ---------------------------------------------------------------------------

_COUNT_STATE_CODES = {0: 0, 1: 1, 2: 2}   # ahead=0, even=1, behind=2
_NUM_SIT = 12   # 3 count × 2 hand × 2 pitch

def _decode_situation_batch(
    codes: torch.Tensor,   # (B,) int in [0, 12]
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose flat situation code into (count_state, hand, pitch_type) indices.
    Code 12 (full-season) is treated as all-zeros for the components.
    """
    # Layout: code = count_idx*4 + hand_idx*2 + pitch_idx
    # Clamp full-season code (12) to 0 — the situation emb will be zeroed by cfg_mask
    c = codes.clone().clamp(0, _NUM_SIT - 1)

    count_idx = c // 4
    rem = c % 4
    hand_idx = rem // 2
    pitch_idx = rem % 2

    return count_idx.to(device), hand_idx.to(device), pitch_idx.to(device)
