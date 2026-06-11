from __future__ import annotations

import numpy as np
import pandas as pd

# Starting values; both are tuned by out-of-sample predictive likelihood in
# the validation stage (Ley et al. 2019 find smooth decay with a half period
# of roughly three years optimal for national teams).
DEFAULT_HALF_LIFE_YEARS = 3.0
DEFAULT_FRIENDLY_WEIGHT = 0.5


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
