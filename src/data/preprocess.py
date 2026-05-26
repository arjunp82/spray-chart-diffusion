"""Convert raw Statcast CSVs into 64×64 spray chart images (.npy files)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------
# Statcast hc_x / hc_y are pixel coords in a ~250×250 top-down field image.
# Home plate sits near (125, 205).  y increases downward in the raw image, so
# we flip it before mapping to our [-1, 1] grid where (0, -1) is home plate
# and (0, +1) is straight-away center field.

HC_X_HOME = 125.42          # home plate x in Statcast pixel space
HC_Y_HOME = 204.5           # home plate y in Statcast pixel space
FIELD_RADIUS_PX = 170.0     # approximate radius covering the outfield wall


def normalize_coords(hc_x: np.ndarray, hc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_norm, y_norm) in [-1, 1] with home plate at (0, 0)."""
    x = (hc_x - HC_X_HOME) / FIELD_RADIUS_PX
    # Flip y so that outfield is positive (upward on plot)
    y = -(hc_y - HC_Y_HOME) / FIELD_RADIUS_PX
    return x, y


def fair_territory_mask(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Boolean mask: True inside the 90-degree fair-territory wedge."""
    # In our normalised system the fair-territory foul lines go at ±45 degrees
    # from the y-axis.  A point is fair if |x| <= y (for y >= 0) or within the
    # small infield area behind home plate.
    angle = np.arctan2(np.abs(grid_x), np.maximum(grid_y, 1e-6))
    return (angle <= np.pi / 4) & (np.sqrt(grid_x ** 2 + grid_y ** 2) <= 1.05)


# Pre-compute a reusable 64×64 fair-territory mask in grid coordinates.
_IMG = 64
_LIN = np.linspace(-1, 1, _IMG)
_GX, _GY = np.meshgrid(_LIN, _LIN[::-1])  # y flipped so index 0 = top of image
FAIR_MASK: np.ndarray = fair_territory_mask(_GX, _GY)


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------
SIGMA_FT = 15.0             # desired kernel sigma in feet
FIELD_RADIUS_FT = 330.0     # approximate radius in feet (matches normalisation)
SIGMA_NORM = SIGMA_FT / FIELD_RADIUS_FT   # sigma in [-1, 1] space
SIGMA_PX = SIGMA_NORM * _IMG / 2          # sigma in pixel units


def coords_to_image(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    """
    Place a Gaussian kernel at each (x_norm, y_norm) on a 64×64 grid,
    sum, apply fair-territory mask, and normalise to sum = 1.
    """
    img = np.zeros((_IMG, _IMG), dtype=np.float32)

    # Map [-1, 1] normalised coords to pixel indices [0, IMG-1]
    px = (x_norm + 1) / 2 * (_IMG - 1)
    # y_norm > 0 is outfield (top of image, index 0); flip
    py = (1 - (y_norm + 1) / 2) * (_IMG - 1)

    for xi, yi in zip(px, py):
        c, r = int(round(xi)), int(round(yi))
        if 0 <= c < _IMG and 0 <= r < _IMG:
            img[r, c] += 1.0

    img = gaussian_filter(img, sigma=SIGMA_PX)
    img *= FAIR_MASK
    total = img.sum()
    if total > 0:
        img /= total
    return img


# ---------------------------------------------------------------------------
# Situation definitions
# ---------------------------------------------------------------------------
PITCH_TYPE_MAP = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball",
    "SL": "offspeed", "CU": "offspeed", "CH": "offspeed",
    "FS": "offspeed", "KC": "offspeed", "EP": "offspeed",
    "ST": "offspeed", "SV": "offspeed",
}

COUNT_STATES = ("ahead", "even", "behind")
HANDEDNESS = ("L", "R")
PITCH_GROUPS = ("fastball", "offspeed")

SITUATION_CODES: dict[str, int] = {}
_idx = 0
for _cs in COUNT_STATES:
    for _hand in HANDEDNESS:
        for _pt in PITCH_GROUPS:
            SITUATION_CODES[f"{_cs}_{_hand}_{_pt}"] = _idx
            _idx += 1
# 12 situations total (3 × 2 × 2)


def count_state(balls: int, strikes: int) -> str:
    if balls > strikes:
        return "ahead"
    if strikes > balls:
        return "behind"
    return "even"


def get_situation_code(cs: str, hand: str, pt_group: str) -> str:
    return f"{cs}_{hand}_{pt_group}"


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(raw_dir.glob("statcast_*.csv")):
        df = pd.read_csv(csv, low_memory=False)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No statcast_*.csv files found in {raw_dir}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} events from {len(frames)} seasons")
    return combined


def assign_split(year: int) -> str:
    if year <= 2021:
        return "train"
    if year == 2022:
        return "val"
    return "test"


def build_full_season_charts(
    df: pd.DataFrame,
    out_dir: Path,
    min_pa: int = 100,
) -> list[dict]:
    records = []
    groups = df.groupby(["batter", "game_year"])
    print(f"\nBuilding full-season charts ({len(groups)} batter-seasons)...")
    for (batter_id, year), grp in tqdm(groups):
        if len(grp) < min_pa:
            continue
        x, y = normalize_coords(grp["hc_x"].values, grp["hc_y"].values)
        img = coords_to_image(x, y)
        rel = f"full_season/{batter_id}_{year}.npy"
        np.save(out_dir / rel, img)
        records.append({
            "file_path": rel,
            "batter_id": int(batter_id),
            "year": int(year),
            "chart_type": "full",
            "situation_code": "full",
            "pa_count": len(grp),
            "split": assign_split(int(year)),
        })
    print(f"  Saved {len(records)} full-season charts")
    return records


def build_situational_charts(
    df: pd.DataFrame,
    out_dir: Path,
    min_pa: int = 30,
) -> list[dict]:
    # Annotate situation columns
    df = df.copy()
    df["pitch_group"] = df["pitch_type"].map(PITCH_TYPE_MAP)
    df["count_state"] = df.apply(
        lambda r: count_state(int(r["balls"]), int(r["strikes"])), axis=1
    )
    # Drop rows where we can't determine the situation
    df = df.dropna(subset=["pitch_group", "p_throws", "count_state"])

    records = []
    groups = df.groupby(["batter", "game_year"])
    print(f"\nBuilding situational charts ({len(groups)} batter-seasons)...")
    for (batter_id, year), grp in tqdm(groups):
        for cs in COUNT_STATES:
            for hand in HANDEDNESS:
                for pt in PITCH_GROUPS:
                    mask = (
                        (grp["count_state"] == cs) &
                        (grp["p_throws"] == hand) &
                        (grp["pitch_group"] == pt)
                    )
                    sub = grp[mask]
                    if len(sub) < min_pa:
                        continue
                    x, y = normalize_coords(sub["hc_x"].values, sub["hc_y"].values)
                    img = coords_to_image(x, y)
                    sit_code = get_situation_code(cs, hand, pt)
                    rel = f"situational/{batter_id}_{year}_{sit_code}.npy"
                    np.save(out_dir / rel, img)
                    records.append({
                        "file_path": rel,
                        "batter_id": int(batter_id),
                        "year": int(year),
                        "chart_type": "situational",
                        "situation_code": sit_code,
                        "pa_count": len(sub),
                        "split": assign_split(int(year)),
                    })
    print(f"  Saved {len(records)} situational charts")
    return records


def build_batter_id_map(df: pd.DataFrame, out_dir: Path) -> dict[int, int]:
    mlbam_ids = sorted(df["batter"].dropna().unique().astype(int).tolist())
    id_map = {mlbam: idx for idx, mlbam in enumerate(mlbam_ids)}
    with open(out_dir / "batter_id_map.json", "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f, indent=2)
    print(f"Saved batter_id_map.json with {len(id_map)} batters")
    return id_map


def preprocess(
    raw_dir: Path,
    processed_dir: Path,
    min_pa_full: int = 100,
    min_pa_sit: int = 30,
) -> None:
    df = load_raw(raw_dir)

    # Drop rows with null landing coordinates or bb_type
    df = df.dropna(subset=["hc_x", "hc_y", "bb_type"])
    df = df[df["events"] != "foul"]
    print(f"After filtering: {len(df):,} batted balls")

    build_batter_id_map(df, processed_dir)

    records: list[dict] = []
    records += build_full_season_charts(df, processed_dir, min_pa=min_pa_full)
    records += build_situational_charts(df, processed_dir, min_pa=min_pa_sit)

    meta = pd.DataFrame(records)
    meta.to_csv(processed_dir / "metadata.csv", index=False)
    print(f"\nMetadata saved: {len(meta)} total charts")
    print(meta["split"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build spray chart images from Statcast CSVs")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-pa-full", type=int, default=100)
    parser.add_argument("--min-pa-sit", type=int, default=30)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.processed_dir, args.min_pa_full, args.min_pa_sit)


if __name__ == "__main__":
    main()
