from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Quadratic penalty pinning mean attack/defense at zero: the standard
# identifiability constraint expressed in a gradient-friendly form.
SUM_PENALTY = 100.0
RHO_BOUNDS = (-0.3, 0.3)
TAU_FLOOR = 1e-10


@dataclass
class DixonColesFit:
    teams: tuple[str, ...]
    mu: float
    host_advantage: float
    rho: float
    attack: np.ndarray
    defense: np.ndarray
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {team: i for i, team in enumerate(self.teams)}

    def rates(
        self, home: str, away: str, host_home: bool = False, host_away: bool = False
    ) -> tuple[float, float]:
        try:
            h, a = self._index[home], self._index[away]
        except KeyError as missing:
            raise KeyError(f"team not in fit: {missing.args[0]!r}") from None
        lam = np.exp(self.mu + self.attack[h] - self.defense[a] + self.host_advantage * host_home)
        nu = np.exp(self.mu + self.attack[a] - self.defense[h] + self.host_advantage * host_away)
        return float(lam), float(nu)

    def score_matrix(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
        max_goals: int = 10,
        rate_scale: float = 1.0,
    ) -> np.ndarray:
        lam, nu = self.rates(home, away, host_home, host_away)
        lam, nu = lam * rate_scale, nu * rate_scale
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, nu))
        corner = np.array(
            [
                [1 - lam * nu * self.rho, 1 + lam * self.rho],
                [1 + nu * self.rho, 1 - self.rho],
            ]
        )
        matrix[:2, :2] *= np.clip(corner, 0.0, None)
        return matrix / matrix.sum()

    def outcome_probs(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
        max_goals: int = 10,
    ) -> tuple[float, float, float]:
        matrix = self.score_matrix(home, away, host_home, host_away, max_goals)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        return home_win, draw, 1.0 - home_win - draw

    def expected_goals(
        self, home: str, away: str, host_home: bool = False, host_away: bool = False
    ) -> tuple[float, float]:
        matrix = self.score_matrix(home, away, host_home, host_away)
        goals = np.arange(matrix.shape[0])
        return float(goals @ matrix.sum(axis=1)), float(goals @ matrix.sum(axis=0))


def _tau_terms(x, y, lam, nu, rho):
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_nu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * nu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + nu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.clip(tau, TAU_FLOOR, None)
    d_lam[m00] = -nu[m00] * rho / tau[m00]
    d_lam[m01] = rho / tau[m01]
    d_nu[m00] = -lam[m00] * rho / tau[m00]
    d_nu[m10] = rho / tau[m10]
    d_rho[m00] = -lam[m00] * nu[m00] / tau[m00]
    d_rho[m01] = lam[m01] / tau[m01]
    d_rho[m10] = nu[m10] / tau[m10]
    d_rho[m11] = -1.0 / tau[m11]
    return np.log(tau), d_lam, d_nu, d_rho


def build_objective(matches: pd.DataFrame, weights: np.ndarray | None = None):
    teams = tuple(sorted(set(matches["home"]) | set(matches["away"])))
    n = len(teams)
    index = {team: i for i, team in enumerate(teams)}
    h = matches["home"].map(index).to_numpy()
    a = matches["away"].map(index).to_numpy()
    x = matches["home_goals"].to_numpy(dtype=float)
    y = matches["away_goals"].to_numpy(dtype=float)
    host = (
        matches["host_home"].to_numpy(dtype=float)
        if "host_home" in matches
        else np.zeros(len(matches))
    )
    w = np.ones(len(matches)) if weights is None else np.asarray(weights, dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu, gamma, rho = theta[0], theta[1], theta[2]
        att = theta[3 : 3 + n]
        dfn = theta[3 + n : 3 + 2 * n]
        lam = np.exp(mu + att[h] - dfn[a] + gamma * host)
        nu = np.exp(mu + att[a] - dfn[h])
        log_tau, d_lam, d_nu, d_rho = _tau_terms(x, y, lam, nu, rho)
        ll = w * (x * np.log(lam) - lam + y * np.log(nu) - nu + log_tau)
        g_eta_h = w * (x - lam + d_lam * lam)
        g_eta_a = w * (y - nu + d_nu * nu)
        att_sum, dfn_sum = att.sum(), dfn.sum()
        value = -ll.sum() + SUM_PENALTY * (att_sum**2 + dfn_sum**2)
        grad = np.concatenate(
            [
                [-(g_eta_h.sum() + g_eta_a.sum())],
                [-(g_eta_h * host).sum()],
                [-(w * d_rho).sum()],
                -(np.bincount(h, g_eta_h, n) + np.bincount(a, g_eta_a, n))
                + 2 * SUM_PENALTY * att_sum,
                (np.bincount(a, g_eta_h, n) + np.bincount(h, g_eta_a, n))
                + 2 * SUM_PENALTY * dfn_sum,
            ]
        )
        return value, grad

    goals_mean = max(float(np.average((x + y) / 2, weights=w)), 0.1)
    theta0 = np.concatenate([[np.log(goals_mean), 0.25, -0.05], np.zeros(2 * n)])
    bounds = [(None, None), (None, None), (RHO_BOUNDS[0], RHO_BOUNDS[1])] + [(None, None)] * (
        2 * n
    )
    return objective, theta0, bounds, teams


def fit_dixon_coles(
    matches: pd.DataFrame, weights: np.ndarray | None = None
) -> DixonColesFit:
    objective, theta0, bounds, teams = build_objective(matches, weights)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Dixon-Coles fit did not converge: {result.message}")
    n = len(teams)
    theta = result.x
    return DixonColesFit(
        teams=teams,
        mu=float(theta[0]),
        host_advantage=float(theta[1]),
        rho=float(theta[2]),
        attack=theta[3 : 3 + n],
        defense=theta[3 + n : 3 + 2 * n],
    )
