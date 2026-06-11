from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cupcast.model.dixon_coles import DixonColesFit, fit_dixon_coles
from cupcast.model.weights import match_weights
from cupcast.validate.metrics import summarize

# Matches with negligible decayed weight only slow the fit down.
WEIGHT_FLOOR = 1e-4


def fit_as_of(
    table: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_years: float,
    friendly_weight: float,
) -> DixonColesFit:
    train = table[table["date"] < as_of].copy()
    train["host_home"] = ~train["neutral"].astype(bool)
    weights = match_weights(
        train["date"], as_of, train["friendly"], half_life_years, friendly_weight
    )
    keep = weights > WEIGHT_FLOOR
    return fit_dixon_coles(train[keep], weights[keep])


def predict_matches(fit: DixonColesFit, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in matches.itertuples():
        try:
            p_home, p_draw, p_away = fit.outcome_probs(
                match.home, match.away, host_home=not match.neutral
            )
        except KeyError:
            continue
        if match.home_goals > match.away_goals:
            outcome = 0
        elif match.home_goals == match.away_goals:
            outcome = 1
        else:
            outcome = 2
        rows.append(
            {
                "date": match.date,
                "home": match.home,
                "away": match.away,
                "tournament": match.tournament,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def prob_array(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
    return probs, predictions["outcome"].to_numpy()


@dataclass(frozen=True)
class Fold:
    train_until: pd.Timestamp
    eval_until: pd.Timestamp


def rolling_folds(start: str, end: str, step_months: int = 6) -> list[Fold]:
    cutoffs = pd.date_range(start=start, end=end, freq=f"{step_months}MS")
    return [
        Fold(train_until=cutoffs[i], eval_until=cutoffs[i + 1])
        for i in range(len(cutoffs) - 1)
    ]


def evaluate_folds(
    table: pd.DataFrame,
    folds: list[Fold],
    half_life_years: float,
    friendly_weight: float,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        fit = fit_as_of(table, fold.train_until, half_life_years, friendly_weight)
        window = table[
            (table["date"] >= fold.train_until) & (table["date"] < fold.eval_until)
        ]
        frames.append(predict_matches(fit, window))
    return pd.concat(frames, ignore_index=True)


def tune_decay(
    table: pd.DataFrame,
    folds: list[Fold],
    half_lives: tuple[float, ...] = (1.5, 2.5, 3.5, 5.0),
    friendly_weights: tuple[float, ...] = (0.3, 0.6, 1.0),
) -> pd.DataFrame:
    rows = []
    for half_life in half_lives:
        for friendly_weight in friendly_weights:
            predictions = evaluate_folds(table, folds, half_life, friendly_weight)
            metrics = summarize(*prob_array(predictions))
            rows.append(
                {"half_life": half_life, "friendly_weight": friendly_weight, **metrics}
            )
    return pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)
