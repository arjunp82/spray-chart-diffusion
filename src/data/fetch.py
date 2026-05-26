"""Download MLB Statcast batted ball data via pybaseball."""

import argparse
import time
from pathlib import Path

import pandas as pd
import pybaseball

pybaseball.cache.enable()

KEEP_COLS = [
    "batter", "pitcher", "hc_x", "hc_y", "launch_speed", "launch_angle",
    "hit_distance_sc", "pitch_type", "balls", "strikes", "stand", "p_throws",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "bb_type", "events", "game_year", "at_bat_number",
]

# These events produce no batted ball landing coordinate.
NON_BATTED = {"strikeout", "walk", "hit_by_pitch", "strikeout_double_play",
              "intent_walk", "catcher_interf"}

SEASON_DATES = {
    2015: ("2015-04-05", "2015-10-04"),
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-28", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
}


def fetch_season(year: int, out_dir: Path) -> pd.DataFrame:
    out_path = out_dir / f"statcast_{year}.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return pd.read_csv(out_path, low_memory=False)

    start, end = SEASON_DATES[year]
    print(f"  Fetching {year}: {start} → {end} ...", flush=True)
    t0 = time.time()
    raw = pybaseball.statcast(start_dt=start, end_dt=end, verbose=False)
    elapsed = time.time() - t0
    print(f"  Raw rows: {len(raw):,}  ({elapsed:.1f}s)")

    # Keep only columns we need (ignore missing ones gracefully)
    available = [c for c in KEEP_COLS if c in raw.columns]
    df = raw[available].copy()

    # Drop rows without landing coordinates (non-batted-ball events)
    df = df.dropna(subset=["hc_x", "hc_y"])

    # Drop foul balls and rows without bb_type (not a fair-territory batted ball)
    df = df[df["events"] != "foul"]
    df = df.dropna(subset=["bb_type"])

    # Drop known non-batted events as a secondary filter
    df = df[~df["events"].isin(NON_BATTED)]

    df["game_year"] = year
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} batted balls → {out_path}")
    return df


def fetch_all(years: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year in sorted(years):
        print(f"\n=== Season {year} ===")
        df = fetch_season(year, out_dir)
        total += len(df)
    print(f"\nDone. Total batted balls across {len(years)} seasons: {total:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Statcast batted ball data")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(SEASON_DATES.keys()),
        help="Which seasons to download (default: 2015–2023)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to write CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    invalid = [y for y in args.years if y not in SEASON_DATES]
    if invalid:
        parser.error(f"Unsupported years: {invalid}. Choose from {list(SEASON_DATES)}")

    fetch_all(args.years, args.out_dir)


if __name__ == "__main__":
    main()
