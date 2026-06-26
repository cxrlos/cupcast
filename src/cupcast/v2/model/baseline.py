"""Static Dixon-Coles baseline via the penaltyblog package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel


def fit_penaltyblog_dc(matches: pd.DataFrame) -> DixonColesGoalModel:
    """Fit a static Dixon-Coles model using penaltyblog.

    Parameters
    ----------
    matches:
        DataFrame with columns ``home``, ``away``, ``home_goals``, ``away_goals``.
        An optional ``weight`` column is forwarded as match weights.

    Returns
    -------
    DixonColesGoalModel
        Fitted penaltyblog model (call ``.predict()`` on it for a fixture).
    """
    weights = (
        matches["weight"].to_numpy(dtype=float).copy() if "weight" in matches.columns else None
    )
    # Copy arrays to ensure writable buffers (penaltyblog's Cython extension
    # rejects read-only memory views from pandas-backed arrays).
    model = DixonColesGoalModel(
        goals_home=matches["home_goals"].to_numpy(dtype=np.int64).copy(),
        goals_away=matches["away_goals"].to_numpy(dtype=np.int64).copy(),
        teams_home=matches["home"].to_numpy().copy(),
        teams_away=matches["away"].to_numpy().copy(),
        weights=weights,
    )
    model.fit()
    return model


def outcome_probs_pb(
    fit: DixonColesGoalModel,
    home: str,
    away: str,
) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) from the penaltyblog DC model."""
    grid = fit.predict(home, away)
    return float(grid.home_win), float(grid.draw), float(grid.away_win)
