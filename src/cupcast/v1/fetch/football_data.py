from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from cupcast.v1.fetch.tls import ensure_system_certificates

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Big-5 European leagues; enough volume for tier-1 engine validation.
DIVISIONS = ["E0", "SP1", "I1", "D1", "F1"]

COLUMNS = {
    "Div": "division",
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    # Pinnacle closing odds: the sharp-book baseline for calibration comparisons.
    "PSCH": "odds_home",
    "PSCD": "odds_draw",
    "PSCA": "odds_away",
}


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_season_csv(
    start_year: int,
    division: str,
    cache_dir: str | Path = "data/raw/football_data",
    session: requests.Session | None = None,
    refresh: bool = False,
) -> str:
    cache_path = Path(cache_dir) / f"{season_code(start_year)}_{division}.csv"
    if cache_path.exists() and not refresh:
        return cache_path.read_text(errors="replace")
    ensure_system_certificates()
    response = (session or requests).get(
        f"{BASE_URL}/{season_code(start_year)}/{division}.csv",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) cupcast/0.1"},
    )
    response.raise_for_status()
    text = response.content.decode("latin-1").lstrip("﻿ï»¿\r\n ")
    first_line = text.splitlines()[0] if text else ""
    if "Div" not in first_line or "HomeTeam" not in first_line:
        redirects = " -> ".join(r.url for r in response.history) or "none"
        raise OSError(
            f"{division} {start_year}: response is not a football-data.co.uk CSV. "
            f"Final URL: {response.url} (redirects: {redirects}); "
            f"starts with: {text[:80]!r}; nothing cached"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def parse_season_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    present = {src: dst for src, dst in COLUMNS.items() if src in raw.columns}
    frame = raw[list(present)].rename(columns=present)
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    return frame.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])


def load_matches(
    start_years: list[int],
    divisions: list[str] = DIVISIONS,
    cache_dir: str | Path = "data/raw/football_data",
) -> pd.DataFrame:
    frames = [
        parse_season_csv(fetch_season_csv(year, division, cache_dir=cache_dir)).assign(
            season=year
        )
        for year in start_years
        for division in divisions
    ]
    return pd.concat(frames, ignore_index=True)


def main(refresh: bool = False) -> None:
    fetched, failed = 0, []
    for year in range(2015, 2026):
        for division in DIVISIONS:
            try:
                fetch_season_csv(year, division, refresh=refresh)
                fetched += 1
            except Exception as exc:  # noqa: BLE001 — keep pulling the rest
                failed.append(f"{division} {year}: {exc}")
    print(f"cached {fetched} season files")
    for line in failed:
        print(f"FAILED {line}")
    if failed and fetched == 0:
        raise OSError(
            f"all {len(failed)} downloads failed; first error: {failed[0]}"
        )
    if failed:
        print("(partial failures above; rerun to retry just those)")


if __name__ == "__main__":
    main()
