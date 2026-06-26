from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from cupcast.v1.fetch.tls import ensure_system_certificates

# Community-maintained CC0 dataset of every men's full international since 1872:
# https://github.com/martj42/international_results
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
EXPECTED_HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"


def fetch_results_csv(
    cache_dir: str | Path = "data/raw/martj42",
    session: requests.Session | None = None,
    refresh: bool = False,
) -> str:
    cache_path = Path(cache_dir) / "results.csv"
    if cache_path.exists() and not refresh:
        return cache_path.read_text()
    ensure_system_certificates()
    response = (session or requests).get(RESULTS_URL, timeout=60)
    response.raise_for_status()
    text = response.text
    if not text.lstrip().startswith(EXPECTED_HEADER):
        raise OSError("unexpected results.csv header; upstream layout changed?")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def load_results(
    cache_dir: str | Path = "data/raw/martj42",
    since: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(fetch_results_csv(cache_dir)), parse_dates=["date"])
    frame = frame.dropna(subset=["home_score", "away_score"])
    frame = frame.rename(
        columns={
            "home_team": "home",
            "away_team": "away",
            "home_score": "home_goals",
            "away_score": "away_goals",
        }
    )
    frame[["home_goals", "away_goals"]] = frame[["home_goals", "away_goals"]].astype(int)
    if since is not None:
        frame = frame[frame["date"] >= pd.Timestamp(since)]
    return frame.reset_index(drop=True)
