from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cupcast.model.dixon_coles import DixonColesFit, fit_dixon_coles
from cupcast.model.weights import match_weights
from cupcast.validate.metrics import summarize

# Matches with negligible decayed weight only slow the fit down.
WEIGHT_FLOOR = 1e-4


def training_set(
    table: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_years: float,
    friendly_weight: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = table[table["date"] < as_of].copy()
    train["host_home"] = ~train["neutral"].astype(bool)
    weights = match_weights(
        train["date"], as_of, train["friendly"], half_life_years, friendly_weight
    )
    keep = weights > WEIGHT_FLOOR
    return train[keep], weights[keep]


def fit_as_of(
    table: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_years: float,
    friendly_weight: float,
) -> DixonColesFit:
    return fit_dixon_coles(*training_set(table, as_of, half_life_years, friendly_weight))


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


def tune_shrinkage(
    table: pd.DataFrame,
    full_table: pd.DataFrame,
    folds: list[Fold],
    ks: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0),
    half_life_years: float = 2.5,
    friendly_weight: float = 1.0,
) -> pd.DataFrame:
    from cupcast.model.shrinkage import (
        apply_shrinkage,
        effective_matches,
        regression_priors,
    )
    from cupcast.ratings.history import elo_history

    prepared = []
    for fold in folds:
        train, weights = training_set(
            table, fold.train_until, half_life_years, friendly_weight
        )
        fit = fit_dixon_coles(train, weights)
        elo, _ = elo_history(full_table[full_table["date"] < fold.train_until])
        n_eff = effective_matches(train, weights, fit.teams)
        priors = regression_priors(fit, n_eff, elo)
        window = table[
            (table["date"] >= fold.train_until) & (table["date"] < fold.eval_until)
        ]
        prepared.append((fit, n_eff, priors, window))

    rows = []
    for k in ks:
        frames = []
        for fit, n_eff, (att_prior, def_prior), window in prepared:
            candidate = (
                apply_shrinkage(fit, n_eff, att_prior, def_prior, k) if k > 0 else fit
            )
            frames.append(predict_matches(candidate, window))
        pooled = pd.concat(frames, ignore_index=True)
        rows.append({"k": k, **summarize(*prob_array(pooled))})
    return pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)


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
