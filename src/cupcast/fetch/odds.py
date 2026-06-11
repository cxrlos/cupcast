from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from cupcast.fetch.tls import ensure_system_certificates

BASE_URL = "https://api.the-odds-api.com/v4"
OUTRIGHT_SPORT = "soccer_fifa_world_cup_winner"
MATCH_SPORT = "soccer_fifa_world_cup"
CACHE_DIR = Path("data/raw/odds_api")
PROCESSED = Path("data/processed")


class OddsApiError(RuntimeError):
    pass


def _key() -> str:
    load_dotenv()
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise OddsApiError("ODDS_API_KEY is not set in .env")
    return key


def fetch_json(path: str, params: dict, cache_name: str, refresh: bool = False) -> list:
    cache_path = CACHE_DIR / f"{cache_name}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())["payload"]
    ensure_system_certificates()
    response = requests.get(
        f"{BASE_URL}/{path}", params={**params, "apiKey": _key()}, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now(UTC).isoformat(), "payload": payload})
    )
    return payload


def implied_probabilities(decimal_odds: list[float]) -> list[float]:
    raw = [1.0 / o for o in decimal_odds]
    total = sum(raw)  # proportional overround removal
    return [p / total for p in raw]


def outright_table(refresh: bool = False) -> pd.DataFrame:
    events = fetch_json(
        f"sports/{OUTRIGHT_SPORT}/odds",
        {"regions": "eu,uk,us", "markets": "outrights", "oddsFormat": "decimal"},
        "outrights",
        refresh,
    )
    rows = []
    for event in events:
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "bookmaker": book["key"],
                            "team": outcome["name"],
                            "decimal_odds": outcome["price"],
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise OddsApiError("no outright odds returned; check sport key or API quota")
    parts = []
    for _, book_frame in frame.groupby("bookmaker"):
        probs = implied_probabilities(book_frame["decimal_odds"].tolist())
        parts.append(book_frame.assign(implied_p=probs))
    table = pd.concat(parts)
    consensus = (
        table.groupby("team")["implied_p"].mean().rename("market_p_champion").reset_index()
    )
    consensus["market_p_champion"] /= consensus["market_p_champion"].sum()
    return consensus.sort_values("market_p_champion", ascending=False, ignore_index=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    consensus = outright_table()
    out = PROCESSED / "market_outrights.csv"
    consensus.to_csv(out, index=False)
    print(f"{len(consensus)} teams -> {out}")
    print(consensus.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
