from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# Understat via soccerdata's maintained reader: plain requests (no browser,
# no Cloudflare), with the session-cookie handshake the site now requires.
# Same metric family as FBref (npxG + xA) over the same big five leagues.
DATA_DIR = Path("data/raw/understat")
PROCESSED = Path("data/processed")
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]
SEASON = "2025-2026"


def to_players_table(stats: pd.DataFrame) -> pd.DataFrame:
    table = stats.reset_index()[["player", "team", "league", "minutes", "np_xg", "xa"]].copy()
    table = table.rename(columns={"np_xg": "npxG", "xa": "xA"})
    for column in ("minutes", "npxG", "xA"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[table["minutes"] > 0]
    table["quality_per90"] = (table["npxG"] + table["xA"]) / (table["minutes"] / 90.0)
    return table


def players_table(refresh: bool = False) -> pd.DataFrame:
    import soccerdata as sd

    reader = sd.Understat(
        leagues=LEAGUES, seasons=SEASON, data_dir=DATA_DIR, no_cache=refresh
    )
    return to_players_table(reader.read_player_season_stats())


def main(refresh: bool = False) -> None:
    warnings.filterwarnings("ignore")
    table = players_table(refresh=refresh)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "understat_players_2025_26.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} player rows across {table['league'].nunique()} leagues -> {out}")


if __name__ == "__main__":
    main()
