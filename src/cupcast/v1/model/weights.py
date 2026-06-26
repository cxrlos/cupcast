from __future__ import annotations

import numpy as np
import pandas as pd

# Tuned by out-of-sample log-loss over eight half-year folds (2022-2026,
# n=4167): optimum flat across 2.5-3.5y half-life, consistent with Ley et al.
# 2019. Friendly down-weighting strictly hurt predictive likelihood at every
# half-life, so friendlies enter at full weight.
DEFAULT_HALF_LIFE_YEARS = 2.5
DEFAULT_FRIENDLY_WEIGHT = 1.0


def match_weights(
    dates: pd.Series,
    as_of: pd.Timestamp,
    friendly: pd.Series,
    half_life_years: float = DEFAULT_HALF_LIFE_YEARS,
    friendly_weight: float = DEFAULT_FRIENDLY_WEIGHT,
) -> np.ndarray:
    age_years = (as_of - dates).dt.days.to_numpy(dtype=float) / 365.25
    decay = np.exp(-np.log(2.0) / half_life_years * age_years)
    return decay * np.where(friendly.to_numpy(), friendly_weight, 1.0)
