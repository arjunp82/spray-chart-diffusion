"""PyTorch Dataset for spray chart images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import (
    COUNT_STATES,
    HANDEDNESS,
    PITCH_GROUPS,
    SITUATION_CODES,
    coords_to_image,
    normalize_coords,
)


class SituationEncoder:
    """Maps situation strings and components to integer codes."""

    # Code 0 is reserved for the unconditional / full-season placeholder.
    FULL_SEASON_CODE = len(SITUATION_CODES)   # == 12

    def encode(self, situation_code: str) -> int:
        if situation_code == "full":
            return self.FULL_SEASON_CODE
        return SITUATION_CODES[situation_code]

    def decode(self, code: int) -> str:
        if code == self.FULL_SEASON_CODE:
            return "full"
        reverse = {v: k for k, v in SITUATION_CODES.items()}
        return reverse[code]

    @property
    def num_codes(self) -> int:
        # 12 situational + 1 full-season
        return len(SITUATION_CODES) + 1


# Scale factor applied to raw probability densities before feeding to the
# diffusion model.  Raw densities have max ≈ 0.008 (way too small for N(0,1)
# noise to carry any signal).  Multiplying by 100 brings peak values to ~0.8,
# which is a reasonable range for DDPM.  Set to 1.0 to reproduce the original
# (broken) behaviour; use 100.0 for all new training runs.
DENSITY_SCALE: float = 100.0


class SprayChartDataset(Dataset):
    """
    Loads processed spray chart .npy images.

    Each item is a dict with:
        image          : FloatTensor [1, 64, 64]  — scaled density
        batter_idx     : int  — embedding index (0 = population prior / unknown)
        situation_code : int  — encoded situation (SituationEncoder)
        chart_type     : str  — 'full' | 'situational'
        pa_count       : int
        partial_image  : FloatTensor [1, 64, 64]  — inpainting conditioning
                         (only if inpaint_mode=True)

    Index 0 in the embedding table is reserved as the population-prior /
    "unknown batter" vector. Held-out batters map to 0 in batter_id_map.json.
    During training, batter_idx is randomly replaced with 0 at rate
    batter_cfg_dropout_prob (analogous to situation CFG dropout), so that
    index 0 learns a real population-level prior rather than being unused.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        chart_type: str | None = None,
        inpaint_mode: bool = False,
        inpaint_k_min: int = 5,
        inpaint_k_max: int = 100,
        raw_events_dir: str | Path | None = None,
        image_scale: float = DENSITY_SCALE,
        batter_cfg_dropout_prob: float = 0.10,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.inpaint_mode = inpaint_mode
        self.inpaint_k_min = inpaint_k_min
        self.inpaint_k_max = inpaint_k_max
        self.image_scale = image_scale
        self.encoder = SituationEncoder()
        # Only apply batter CFG dropout during training
        self.batter_cfg_dropout_prob = batter_cfg_dropout_prob if split == "train" else 0.0

        # Load batter ID map
        id_map_path = self.processed_dir / "batter_id_map.json"
        with open(id_map_path) as f:
            raw_map = json.load(f)
        self.batter_id_map: dict[int, int] = {int(k): v for k, v in raw_map.items()}

        # Load and filter metadata
        meta = pd.read_csv(self.processed_dir / "metadata.csv")
        meta = meta[meta["split"] == split].reset_index(drop=True)
        if chart_type is not None:
            meta = meta[meta["chart_type"] == chart_type].reset_index(drop=True)
        self.meta = meta

        # Optional: raw event CSV directory for building partial charts on the fly
        self.raw_events: dict[int, pd.DataFrame] | None = None
        if inpaint_mode and raw_events_dir is not None:
            self.raw_events = self._load_raw_events(Path(raw_events_dir))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        img_path = self.processed_dir / row["file_path"]
        image = np.load(img_path).astype(np.float32)                    # (64, 64)
        image_t = torch.from_numpy(image).unsqueeze(0) * self.image_scale  # (1, 64, 64)

        batter_mlbam = int(row["batter_id"])
        batter_idx = self.batter_id_map.get(batter_mlbam, 0)
        # Batter CFG dropout: replace personal embedding with population prior (0)
        # so index 0 learns a real batter prior, not a meaningless unused vector.
        if self.batter_cfg_dropout_prob > 0 and np.random.random() < self.batter_cfg_dropout_prob:
            batter_idx = 0
        sit_code = self.encoder.encode(str(row["situation_code"]))

        item = {
            "image": image_t,
            "batter_idx": batter_idx,
            "situation_code": sit_code,
            "chart_type": str(row["chart_type"]),
            "pa_count": int(row["pa_count"]),
        }

        if self.inpaint_mode:
            partial = self._build_partial(batter_mlbam, row, full_image=image)
            item["partial_image"] = partial

        return item

    # ------------------------------------------------------------------
    def _build_partial(
        self,
        batter_mlbam: int,
        row: pd.Series,
        full_image: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Build a partial spray chart from k randomly sampled events.

        If raw CSV events are available, samples directly from those.
        Otherwise synthesises a partial by drawing k pixel positions
        from the full density image (proportional to density values),
        building a normalised histogram.  This keeps the partial
        in-distribution with the training target even when raw CSVs
        are absent (which is the common case).
        """
        k = np.random.randint(self.inpaint_k_min, self.inpaint_k_max + 1)

        # --- Path 1: raw events available ---
        if self.raw_events is not None and batter_mlbam in self.raw_events:
            events = self.raw_events[batter_mlbam]
            if len(events) > 0:
                k_actual = min(k, len(events))
                sampled = events.sample(n=k_actual, replace=False)
                x, y = normalize_coords(sampled["hc_x"].values, sampled["hc_y"].values)
                partial = coords_to_image(x, y)
                return torch.from_numpy(partial).unsqueeze(0) * self.image_scale

        # --- Path 2: synthesise from the full density image ---
        if full_image is None or full_image.sum() < 1e-12:
            return torch.zeros(1, 64, 64)

        flat = full_image.ravel().astype(np.float64)
        flat = np.maximum(flat, 0.0)
        total = flat.sum()
        if total < 1e-12:
            return torch.zeros(1, 64, 64)
        probs = flat / total

        sampled_idx = np.random.choice(len(flat), size=k, replace=True, p=probs)
        partial_flat = np.bincount(sampled_idx, minlength=len(flat)).astype(np.float32)
        partial_flat /= partial_flat.sum() + 1e-12
        partial = partial_flat.reshape(full_image.shape)
        return torch.from_numpy(partial).unsqueeze(0) * self.image_scale

    @staticmethod
    def _load_raw_events(raw_dir: Path) -> dict[int, pd.DataFrame]:
        frames: dict[int, list[pd.DataFrame]] = {}
        for csv in sorted(raw_dir.glob("statcast_*.csv")):
            df = pd.read_csv(csv, low_memory=False, usecols=["batter", "hc_x", "hc_y"])
            df = df.dropna(subset=["hc_x", "hc_y"])
            for batter_id, grp in df.groupby("batter"):
                bid = int(batter_id)
                frames.setdefault(bid, []).append(grp[["hc_x", "hc_y"]])
        return {bid: pd.concat(dfs, ignore_index=True) for bid, dfs in frames.items()}
