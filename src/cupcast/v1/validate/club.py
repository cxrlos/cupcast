from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cupcast.v1.fetch.football_data import DIVISIONS, load_matches
from cupcast.v1.model.dixon_coles import fit_dixon_coles
from cupcast.v1.model.weights import match_weights
from cupcast.v1.validate.metrics import summarize

CLUB_DATA_DIR = Path("data/raw/football_data")
EVAL_START = pd.Timestamp("2017-08-01")
REFIT_DAYS = 30


def club_data_available() -> bool:
    return CLUB_DATA_DIR.exists() and any(CLUB_DATA_DIR.glob("*.csv"))


def _prepare_division(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["friendly"] = False
    frame["host_home"] = True  # club football: listed home side is at home
    return frame


def rolling_club_predictions(
    division_frame: pd.DataFrame, half_life_years: float = 2.5
) -> pd.DataFrame:
    frame = _prepare_division(division_frame)
    rows = []
    cutoff = max(EVAL_START, frame["date"].min() + pd.Timedelta(days=3 * 365))
    end = frame["date"].max()
    while cutoff <= end:
        train = frame[frame["date"] < cutoff]
        window = frame[
            (frame["date"] >= cutoff) & (frame["date"] < cutoff + pd.Timedelta(days=REFIT_DAYS))
        ]
        if len(train) >= 500 and not window.empty:
            weights = match_weights(
                train["date"], cutoff, train["friendly"], half_life_years, 1.0
            )
            fit = fit_dixon_coles(train, weights)
            for match in window.itertuples():
                if match.home not in fit.teams or match.away not in fit.teams:
                    continue
                p_home, p_draw, p_away = fit.outcome_probs(match.home, match.away, host_home=True)
                rows.append(
                    {
                        "date": match.date,
                        "division": match.division,
                        "p_home": p_home,
                        "p_draw": p_draw,
                        "p_away": p_away,
                        "odds_home": match.odds_home,
                        "odds_draw": match.odds_draw,
                        "odds_away": match.odds_away,
                        "outcome": 0
                        if match.home_goals > match.away_goals
                        else (1 if match.home_goals == match.away_goals else 2),
                    }
                )
        cutoff += pd.Timedelta(days=REFIT_DAYS)
    return pd.DataFrame(rows)


def market_probs(predictions: pd.DataFrame) -> np.ndarray:
    inverse = 1.0 / predictions[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    return inverse / inverse.sum(axis=1, keepdims=True)


def club_backtest(start_years: list[int] | None = None) -> pd.DataFrame:
    start_years = start_years or list(range(2015, 2026))
    data = load_matches(start_years)
    frames = []
    for division in DIVISIONS:
        division_frame = data[data["division"] == division]
        if division_frame.empty:
            continue
        frames.append(rolling_club_predictions(division_frame))
    predictions = pd.concat(frames, ignore_index=True)
    return predictions


def club_report(predictions: pd.DataFrame) -> pd.DataFrame:
    has_odds = predictions.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    model = summarize(
        has_odds[["p_home", "p_draw", "p_away"]].to_numpy(), has_odds["outcome"].to_numpy()
    )
    market = summarize(market_probs(has_odds), has_odds["outcome"].to_numpy())
    return pd.DataFrame(
        [
            {"forecaster": "Dixon-Coles", **model},
            {"forecaster": "Pinnacle closing", **market},
        ]
    )
