from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

from cupcast.fetch.tls import ensure_system_certificates

# Understat embeds the full player-season table as a JSON literal in each
# league page — plain requests, no browser, no challenge. Covers exactly the
# big five leagues the squad composite uses; metrics are npxG and xA, the
# same family as the FBref composite.
BASE_URL = "https://understat.com/league"
LEAGUES = ("EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1")
SEASON = "2025"  # Understat labels seasons by their starting year
PROCESSED = Path("data/processed")

PLAYERS_RE = re.compile(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", re.DOTALL)


def fetch_league_players(
    league: str,
    season: str = SEASON,
    cache_dir: str | Path = "data/raw/understat",
    session: requests.Session | None = None,
    refresh: bool = False,
) -> list[dict]:
    cache_path = Path(cache_dir) / f"{league}_{season}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    ensure_system_certificates()
    response = (session or requests).get(
        f"{BASE_URL}/{league}/{season}",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) cupcast/0.1"},
    )
    response.raise_for_status()
    players = parse_players_payload(response.text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(players))
    return players


def parse_players_payload(html: str) -> list[dict]:
    match = PLAYERS_RE.search(html)
    if match is None:
        raise OSError("playersData blob not found; Understat page layout changed?")
    decoded = match.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(decoded)


def players_table(season: str = SEASON, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for league in LEAGUES:
        players = fetch_league_players(league, season, refresh=refresh)
        frame = pd.DataFrame(players)
        frame["league"] = league
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    for column in ("time", "npxG", "xA"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.rename(columns={"player_name": "player", "time": "minutes"})
    table["quality_per90"] = (table["npxG"] + table["xA"]) / (table["minutes"] / 90.0)
    return table[["player", "team_title", "league", "minutes", "npxG", "xA", "quality_per90"]]


def main(refresh: bool = False) -> None:
    table = players_table(refresh=refresh)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "understat_players_2025_26.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} player rows across {table['league'].nunique()} leagues -> {out}")


if __name__ == "__main__":
    main()
