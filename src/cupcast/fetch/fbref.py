from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/fbref")
PROCESSED = Path("data/processed")
LEAGUES = "Big 5 European Leagues Combined"
SEASON = "2025-2026"
# standard covers minutes, xG, npxG, xAG and progressive actions; defense adds
# tackles/interceptions; keeper_adv adds PSxG +/- for shootout adjustments.
STAT_TYPES = ("standard", "defense", "keeper_adv")


def pull_player_season_stats() -> dict[str, pd.DataFrame]:
    # FBref sits behind a Cloudflare challenge, so soccerdata drives a real
    # browser. Endpoint-security tooling on managed machines kills the
    # bundled driver: run this on an unmanaged machine, then sync data/.
    import soccerdata as sd

    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASON, data_dir=DATA_DIR)
    frames = {}
    for stat_type in STAT_TYPES:
        frames[stat_type] = fbref.read_player_season_stats(stat_type=stat_type)
    return frames


def main() -> None:
    warnings.filterwarnings("ignore")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for stat_type, frame in pull_player_season_stats().items():
        out = PROCESSED / f"fbref_{stat_type}_2025_26.csv"
        frame.reset_index().to_csv(out, index=False)
        print(f"{stat_type}: {frame.shape[0]} player rows -> {out}")


if __name__ == "__main__":
    main()
