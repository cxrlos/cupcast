"""
Synthetic-data recovery tests for dynamic_dc_model.

Generates match data where one team's attack trends upward across quarters,
fits the model via SVI, and asserts:
  - the recovered attack trajectory increases for the trending team
  - sigma_att is positive in the posterior

A separate smoke test verifies shapes when attack_prior_loc is provided.

JAX is forced to CPU to keep wall-time predictable.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import numpyro
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from cupcast.v2.model.dynamic import dynamic_dc_model

numpyro.set_host_device_count(1)

# ---------------------------------------------------------------------------
# Synthetic data — one trending team, rest flat
# ---------------------------------------------------------------------------

N_TEAMS = 5
N_PERIODS = 8
TRENDING_TEAM = 0

_RNG = np.random.default_rng(2026)

_MU = 0.0
_DEFENSE_TRUE = np.zeros(N_TEAMS)

# Team 0 attack goes 0 → 2.0 linearly; teams 1-4 flat at 0.
_ATTACK_TRUE = np.zeros((N_TEAMS, N_PERIODS))
for _t in range(N_PERIODS):
    _ATTACK_TRUE[TRENDING_TEAM, _t] = 2.0 * _t / (N_PERIODS - 1)

_PAIRS = [(h, a) for h in range(N_TEAMS) for a in range(N_TEAMS) if h != a]

_home_idx_list: list[int] = []
_away_idx_list: list[int] = []
_period_list: list[int] = []
_x_list: list[int] = []
_y_list: list[int] = []
_host_list: list[float] = []

for _p in range(N_PERIODS):
    for _h, _a in _PAIRS:
        lam = np.exp(_MU + _ATTACK_TRUE[_h, _p] - _DEFENSE_TRUE[_a])
        nu = np.exp(_MU + _ATTACK_TRUE[_a, _p] - _DEFENSE_TRUE[_h])
        _x_list.append(int(_RNG.poisson(lam)))
        _y_list.append(int(_RNG.poisson(nu)))
        _home_idx_list.append(_h)
        _away_idx_list.append(_a)
        _period_list.append(_p)
        _host_list.append(0.0)

_HOME_IDX = np.array(_home_idx_list, dtype=np.int32)
_AWAY_IDX = np.array(_away_idx_list, dtype=np.int32)
_PERIOD = np.array(_period_list, dtype=np.int32)
_HOST = np.array(_host_list, dtype=float)
_X = np.array(_x_list, dtype=np.int32)
_Y = np.array(_y_list, dtype=np.int32)

_DATA_KWARGS = dict(
    home_idx=_HOME_IDX,
    away_idx=_AWAY_IDX,
    period=_PERIOD,
    host=_HOST,
    x=_X,
    y=_Y,
    n_teams=N_TEAMS,
    n_periods=N_PERIODS,
)


# ---------------------------------------------------------------------------
# Trend recovery and sigma tests (shared SVI run via setup_class)
# ---------------------------------------------------------------------------


class TestDynamicDcTrendRecovery:
    """Fit dynamic_dc_model on synthetic trending data and verify recovery."""

    @classmethod
    def setup_class(cls):
        guide = AutoNormal(dynamic_dc_model)
        svi = SVI(dynamic_dc_model, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
        result = svi.run(random.PRNGKey(0), 2500, progress_bar=False, **_DATA_KWARGS)

        # Model Predictive gives deterministic sites (attack_traj, attack_now, defense_now).
        model_pred = Predictive(
            dynamic_dc_model, guide=guide, params=result.params, num_samples=200
        )
        cls.model_samples = model_pred(random.PRNGKey(1), **_DATA_KWARGS)

        # Guide Predictive gives latent sites (sigma_att, mu, rho, …).
        guide_pred = Predictive(guide, params=result.params, num_samples=200)
        cls.guide_samples = guide_pred(random.PRNGKey(2), **_DATA_KWARGS)

    def test_attack_trend_recovered(self):
        """Posterior mean attack for the trending team must rise from first to last period."""
        attack_traj = np.array(self.model_samples["attack_traj"])  # (200, n_teams, n_periods)
        att_mean = attack_traj.mean(axis=0)  # (n_teams, n_periods)

        first = float(att_mean[TRENDING_TEAM, 0])
        last = float(att_mean[TRENDING_TEAM, -1])
        assert last > first, (
            f"Trending team attack did not increase: "
            f"period-0 mean={first:.3f}, period-7 mean={last:.3f}"
        )

    def test_sigma_att_positive(self):
        """Posterior sigma_att must be positive (random-walk dynamics are non-degenerate)."""
        sigma_att = np.array(self.guide_samples["sigma_att"])
        assert float(sigma_att.mean()) > 0.0, (
            f"sigma_att posterior mean should be positive, got {float(sigma_att.mean()):.4f}"
        )


# ---------------------------------------------------------------------------
# Smoke test — shape check with attack_prior_loc
# ---------------------------------------------------------------------------


class TestDynamicDcSmoke:
    """Shape / interface smoke test; does not check parameter recovery."""

    def test_attack_now_shape_with_prior(self):
        """attack_now must have shape (n_teams,) when attack_prior_loc is supplied."""
        n_teams = 4
        n_periods = 3

        home_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        away_idx = np.array([1, 2, 3, 0], dtype=np.int32)
        period = np.array([0, 1, 2, 2], dtype=np.int32)
        host = np.array([1.0, 0.0, 1.0, 0.0])
        x = np.array([1, 2, 0, 1], dtype=np.int32)
        y = np.array([0, 1, 2, 0], dtype=np.int32)
        attack_prior_loc = np.array([0.5, -0.2, 0.1, -0.3])

        guide = AutoNormal(dynamic_dc_model)
        svi = SVI(dynamic_dc_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        result = svi.run(
            random.PRNGKey(0),
            10,
            home_idx=home_idx,
            away_idx=away_idx,
            period=period,
            host=host,
            x=x,
            y=y,
            n_teams=n_teams,
            n_periods=n_periods,
            attack_prior_loc=attack_prior_loc,
            progress_bar=False,
        )

        predictive = Predictive(
            dynamic_dc_model, guide=guide, params=result.params, num_samples=5
        )
        samples = predictive(
            random.PRNGKey(1),
            home_idx=home_idx,
            away_idx=away_idx,
            period=period,
            host=host,
            x=x,
            y=y,
            n_teams=n_teams,
            n_periods=n_periods,
            attack_prior_loc=attack_prior_loc,
        )

        attack_now = np.array(samples["attack_now"])
        assert attack_now.shape == (5, n_teams), (
            f"Expected (5, {n_teams}), got {attack_now.shape}"
        )
