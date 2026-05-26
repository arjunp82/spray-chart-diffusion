"""Four baselines for spray chart prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

from src.data.preprocess import FAIR_MASK, normalize_coords, coords_to_image

_IMG = 64
_EPS = 1e-10

# Grid in normalised space — shared by all baselines
_LIN = np.linspace(-1, 1, _IMG)
_GX, _GY = np.meshgrid(_LIN, _LIN[::-1])
_PTS = np.stack([_GX.ravel(), _GY.ravel()], axis=0)    # (2, 4096)


def _density_from_kde(kde: gaussian_kde) -> np.ndarray:
    """Evaluate kde on the 64×64 grid, apply fair-territory mask, normalise."""
    density = kde(_PTS).reshape(_IMG, _IMG).astype(np.float32)
    density *= FAIR_MASK
    total = density.sum()
    return density / total if total > 0 else np.full((_IMG, _IMG), 1.0 / (_IMG * _IMG))


def _uniform_density() -> np.ndarray:
    d = FAIR_MASK.astype(np.float32)
    return d / d.sum()


class BaselinePredictor(ABC):
    @abstractmethod
    def fit(self, events_df) -> None: ...

    @abstractmethod
    def predict(self, batter_id: int, situation: dict) -> np.ndarray:
        """Returns a 64×64 normalised density image."""


# ---------------------------------------------------------------------------
# 1. Historical KDE — career average, no situational adjustment
# ---------------------------------------------------------------------------

class HistoricalKDE(BaselinePredictor):
    """Career average spray chart using all batted ball coords."""

    def __init__(self) -> None:
        self._kdes: dict[int, gaussian_kde] = {}

    def fit(self, events_df) -> None:
        for batter_id, grp in events_df.groupby("batter"):
            x, y = normalize_coords(grp["hc_x"].values, grp["hc_y"].values)
            pts = np.stack([x, y], axis=0)
            if pts.shape[1] >= 2:
                try:
                    self._kdes[int(batter_id)] = gaussian_kde(pts)
                except np.linalg.LinAlgError:
                    pass

    def predict(self, batter_id: int, situation: dict) -> np.ndarray:
        kde = self._kdes.get(batter_id)
        if kde is None:
            return _uniform_density()
        return _density_from_kde(kde)


# ---------------------------------------------------------------------------
# 2. Platoon KDE — separate densities by pitcher handedness
# ---------------------------------------------------------------------------

class PlatoonKDE(BaselinePredictor):
    """Separate KDEs for vs. LHP and vs. RHP."""

    def __init__(self) -> None:
        # key: (batter_id, hand) where hand in ('L', 'R')
        self._kdes: dict[tuple[int, str], gaussian_kde] = {}

    def fit(self, events_df) -> None:
        for (batter_id, hand), grp in events_df.groupby(["batter", "p_throws"]):
            x, y = normalize_coords(grp["hc_x"].values, grp["hc_y"].values)
            pts = np.stack([x, y], axis=0)
            if pts.shape[1] >= 2:
                try:
                    self._kdes[(int(batter_id), str(hand))] = gaussian_kde(pts)
                except np.linalg.LinAlgError:
                    pass

    def predict(self, batter_id: int, situation: dict) -> np.ndarray:
        hand = situation.get("p_throws", "R")
        kde = self._kdes.get((batter_id, hand))
        if kde is None:
            # Fall back to opposite hand or uniform
            other = "L" if hand == "R" else "R"
            kde = self._kdes.get((batter_id, other))
        if kde is None:
            return _uniform_density()
        return _density_from_kde(kde)


# ---------------------------------------------------------------------------
# 3. Situational KDE — exact situation filter
# ---------------------------------------------------------------------------

class SituationalKDE(BaselinePredictor):
    """
    KDE restricted to (count_state, pitcher_hand, pitch_type_group).
    Falls back to uniform if fewer than 10 events.
    """

    MIN_EVENTS = 10

    def __init__(self) -> None:
        self._kdes: dict[tuple, gaussian_kde] = {}

    def fit(self, events_df) -> None:
        from src.data.preprocess import PITCH_TYPE_MAP, count_state as _count_state

        df = events_df.copy()
        df["pitch_group"] = df["pitch_type"].map(PITCH_TYPE_MAP)
        df["count_state"] = df.apply(
            lambda r: _count_state(int(r["balls"]), int(r["strikes"])), axis=1
        )
        df = df.dropna(subset=["pitch_group", "p_throws", "count_state"])

        for (batter_id, cs, hand, pg), grp in df.groupby(
            ["batter", "count_state", "p_throws", "pitch_group"]
        ):
            if len(grp) < self.MIN_EVENTS:
                continue
            x, y = normalize_coords(grp["hc_x"].values, grp["hc_y"].values)
            pts = np.stack([x, y], axis=0)
            try:
                self._kdes[(int(batter_id), cs, hand, pg)] = gaussian_kde(pts)
            except np.linalg.LinAlgError:
                pass

    def predict(self, batter_id: int, situation: dict) -> np.ndarray:
        cs = situation.get("count_state", "even")
        hand = situation.get("p_throws", "R")
        pg = situation.get("pitch_group", "fastball")
        kde = self._kdes.get((batter_id, cs, hand, pg))
        if kde is None:
            return _uniform_density()
        return _density_from_kde(kde)


# ---------------------------------------------------------------------------
# 4. Discriminative CNN — point estimate, no uncertainty
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DiscriminativeCNN(nn.Module, BaselinePredictor):
    """
    ~5M-parameter CNN that predicts a 64×64 spray chart from
    batter embedding + situational features.  No diffusion, no uncertainty.
    """

    def __init__(self, num_batters: int, cond_dim: int = 128) -> None:
        nn.Module.__init__(self)
        self.batter_emb = nn.Embedding(num_batters, cond_dim)
        # Situational: count(3) + hand(2) + pitch(2) → one-hot dim 7
        self.sit_proj = nn.Linear(7, cond_dim)
        self.cond_proj = nn.Linear(2 * cond_dim, cond_dim)

        # Upsampling backbone: start from 4×4 feature map
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(cond_dim, 256, 4),            # 4×4
            nn.ReLU(inplace=True),
            _ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # 8×8
            nn.ReLU(inplace=True),
            _ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),    # 16×16
            nn.ReLU(inplace=True),
            _ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # 32×32
            nn.ReLU(inplace=True),
            _ResBlock(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),     # 64×64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(
        self,
        batter_idx: torch.Tensor,   # (B,)
        count_state: torch.Tensor,  # (B,) int {0,1,2}
        hand: torch.Tensor,         # (B,) int {0,1}
        pitch: torch.Tensor,        # (B,) int {0,1}
    ) -> torch.Tensor:
        bat_e = self.batter_emb(batter_idx)                        # (B, D)
        sit_oh = torch.cat([
            torch.nn.functional.one_hot(count_state, 3).float(),
            torch.nn.functional.one_hot(hand, 2).float(),
            torch.nn.functional.one_hot(pitch, 2).float(),
        ], dim=-1)
        sit_e = self.sit_proj(sit_oh)
        cond = self.cond_proj(torch.cat([bat_e, sit_e], dim=-1))   # (B, D)
        x = self.backbone(cond.unsqueeze(-1).unsqueeze(-1))        # (B, 1, 64, 64)
        fair = torch.from_numpy(FAIR_MASK.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = torch.relu(x) * fair.to(x.device)
        x = x / (x.sum(dim=(-2, -1), keepdim=True) + _EPS)
        return x

    # BaselinePredictor interface (inference-only, after training)
    def fit(self, events_df) -> None:
        raise NotImplementedError("Use train.py to train the discriminative CNN.")

    def predict(self, batter_id: int, situation: dict) -> np.ndarray:
        raise NotImplementedError("Call forward() with tensor inputs directly.")
