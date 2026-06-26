"""DC-corrected score-matrix and outcome predictions from a Posterior."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from cupcast.v2.model.fit import Posterior


def score_matrix(
    posterior: Posterior,
    home: str,
    away: str,
    host_home: bool,
    max_goals: int = 10,
    rate_scale: float = 1.0,
) -> np.ndarray:
    """DC-corrected bivariate-Poisson score matrix using posterior-mean rates.

    Returns a normalized ``(max_goals+1, max_goals+1)`` matrix where
    ``matrix[i, j]`` = P(home scores i, away scores j).
    """
    lam, nu = posterior.rate(home, away, host_home)
    lam, nu = lam * rate_scale, nu * rate_scale
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, nu))
    corner = np.array(
        [
            [1 - lam * nu * posterior.rho, 1 + lam * posterior.rho],
            [1 + nu * posterior.rho, 1 - posterior.rho],
        ]
    )
    matrix[:2, :2] *= np.clip(corner, 0.0, None)
    return matrix / matrix.sum()


def outcome_probs(
    posterior: Posterior,
    home: str,
    away: str,
    host_home: bool,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win) from the DC score matrix."""
    matrix = score_matrix(posterior, home, away, host_home, max_goals)
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    return p_home, p_draw, 1.0 - p_home - p_draw


def expected_goals(
    posterior: Posterior,
    home: str,
    away: str,
    host_home: bool,
) -> tuple[float, float]:
    """Return (expected home goals, expected away goals) from the score matrix."""
    matrix = score_matrix(posterior, home, away, host_home)
    goals = np.arange(matrix.shape[0])
    return float(goals @ matrix.sum(axis=1)), float(goals @ matrix.sum(axis=0))
