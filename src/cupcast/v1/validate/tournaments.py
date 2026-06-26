from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v1.validate.backtest import fit_as_of, predict_matches, prob_array
from cupcast.v1.validate.metrics import summarize

# Backtest tournaments: train strictly before the opening match.
REPLAYS = {
    "euro2024": {"tournament": "UEFA Euro", "start": "2024-06-14"},
    "copa2024": {"tournament": "Copa América", "start": "2024-06-20"},
}


def replay_tournament(
    table: pd.DataFrame,
    key: str,
    half_life_years: float,
    friendly_weight: float,
) -> pd.DataFrame:
    spec = REPLAYS[key]
    start = pd.Timestamp(spec["start"])
    fit = fit_as_of(table, start, half_life_years, friendly_weight)
    matches = table[
        (table["tournament"] == spec["tournament"])
        & (table["date"] >= start)
        & (table["date"] < start + pd.Timedelta(days=45))
    ]
    return predict_matches(fit, matches)


def replay_report(
    table: pd.DataFrame, half_life_years: float, friendly_weight: float
) -> pd.DataFrame:
    rows = []
    for key in REPLAYS:
        predictions = replay_tournament(table, key, half_life_years, friendly_weight)
        probs, outcome = prob_array(predictions)
        metrics = summarize(probs, outcome)
        uniform = np.full_like(probs, 1 / 3)
        baseline = summarize(uniform, outcome)
        rows.append(
            {
                "tournament": key,
                **metrics,
                "log_loss_uniform": baseline["log_loss"],
                "rps_uniform": baseline["rps"],
            }
        )
    return pd.DataFrame(rows)
