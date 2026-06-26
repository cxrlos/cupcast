"""
Tests for the clubform-informed learned prior (f_θ).

Covers:
  1. clubform_prior_locs — alignment, standardization, and missing-team handling.
  2. dynamic_dc_with_prior — synthetic recovery: assert a1 > 0 (clubform predicts
     attack) and that a data-poor team's posterior attack is anchored to its
     clubform-implied prior rather than the uninformative 0.

JAX is forced to CPU to keep wall-time predictable.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import numpyro
import pandas as pd
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from cupcast.v2.model.prior import clubform_prior_locs, dynamic_dc_with_prior

numpyro.set_host_device_count(1)


# ---------------------------------------------------------------------------
# Unit tests for clubform_prior_locs
# ---------------------------------------------------------------------------


class TestClubformPriorLocs:
    """Alignment, standardization, and missing-team behaviour."""

    # Clubform for three teams; "D" is intentionally absent.
    _CF = pd.DataFrame(
        {
            "team": ["A", "B", "C"],
            "attack": [3.0, 1.0, 2.0],
            "defense": [2.0, 3.0, 1.0],
            "gk": [0.5, 0.5, 0.5],
            "coverage": [1.0, 1.0, 1.0],
        }
    )
    _TEAMS = ("A", "B", "C", "D")

    def _locs(self):
        return clubform_prior_locs(self._CF, self._TEAMS)

    def test_output_length(self):
        att, dfn = self._locs()
        assert len(att) == len(self._TEAMS)
        assert len(dfn) == len(self._TEAMS)

    def test_alignment_order(self):
        # attack: vals=[3,1,2], mean=2, std=1  → A=1, B=-1, C=0
        # defense: vals=[2,3,1], mean=2, std=1 → A=0, B=1, C=-1
        att, dfn = self._locs()
        np.testing.assert_allclose(att[:3], [1.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(dfn[:3], [0.0, 1.0, -1.0], atol=1e-9)

    def test_standardization_mean_near_zero(self):
        att, dfn = self._locs()
        # Mean over the three teams that have clubform should be ≈ 0.
        assert abs(att[:3].mean()) < 1e-9
        assert abs(dfn[:3].mean()) < 1e-9

    def test_missing_team_is_zero(self):
        att, dfn = self._locs()
        assert att[3] == 0.0
        assert dfn[3] == 0.0


# ---------------------------------------------------------------------------
# Synthetic data for dynamic_dc_with_prior recovery
# ---------------------------------------------------------------------------

N_TEAMS = 6
N_PERIODS = 4
POOR_TEAM = 0  # gets only 2 matches total

# Pre-standardised clubform values (mean≈0 by construction).
CF_ATT = np.array([2.0, 1.5, 0.5, -0.5, -1.5, -2.0], dtype=float)
CF_DEF = np.array([1.0, 0.8, 0.2, -0.2, -0.8, -1.0], dtype=float)

# Ground truth: attack0[i] = TRUE_A1 * CF_ATT[i], no defence signal, no walk.
_TRUE_A1 = 0.8
_TRUE_MU = 0.0
_att0_true = _TRUE_A1 * CF_ATT
_def0_true = np.zeros(N_TEAMS)

_RNG = np.random.default_rng(2026)

_home_idx_list: list[int] = []
_away_idx_list: list[int] = []
_period_list: list[int] = []
_x_list: list[int] = []
_y_list: list[int] = []
_host_list: list[float] = []

# Data-rich teams (1–5): every ordered pair, every period.
_rich_pairs = [(h, a) for h in range(1, N_TEAMS) for a in range(1, N_TEAMS) if h != a]
for _p in range(N_PERIODS):
    for _h, _a in _rich_pairs:
        _lam = np.exp(_TRUE_MU + _att0_true[_h] - _def0_true[_a])
        _nu = np.exp(_TRUE_MU + _att0_true[_a] - _def0_true[_h])
        _home_idx_list.append(_h)
        _away_idx_list.append(_a)
        _period_list.append(_p)
        _x_list.append(int(_RNG.poisson(_lam)))
        _y_list.append(int(_RNG.poisson(_nu)))
        _host_list.append(0.0)

# Data-poor team (0): only 2 matches, period 0.
for _h, _a in [(POOR_TEAM, 1), (2, POOR_TEAM)]:
    _lam = np.exp(_TRUE_MU + _att0_true[_h] - _def0_true[_a])
    _nu = np.exp(_TRUE_MU + _att0_true[_a] - _def0_true[_h])
    _home_idx_list.append(_h)
    _away_idx_list.append(_a)
    _period_list.append(0)
    _x_list.append(int(_RNG.poisson(_lam)))
    _y_list.append(int(_RNG.poisson(_nu)))
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
    clubform_attack=CF_ATT,
    clubform_defense=CF_DEF,
)


# ---------------------------------------------------------------------------
# Synthetic recovery and anchoring tests
# ---------------------------------------------------------------------------


class TestDynamicDcWithPriorRecovery:
    """Fit dynamic_dc_with_prior on synthetic data and assert f_θ recovery."""

    _N_SAMPLES = 200

    @classmethod
    def setup_class(cls):
        guide = AutoNormal(dynamic_dc_with_prior)
        svi = SVI(dynamic_dc_with_prior, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
        result = svi.run(random.PRNGKey(0), 2500, progress_bar=False, **_DATA_KWARGS)

        model_pred = Predictive(
            dynamic_dc_with_prior, guide=guide, params=result.params, num_samples=cls._N_SAMPLES
        )
        cls.model_samples = model_pred(random.PRNGKey(1), **_DATA_KWARGS)

        guide_pred = Predictive(guide, params=result.params, num_samples=cls._N_SAMPLES)
        cls.guide_samples = guide_pred(random.PRNGKey(2), **_DATA_KWARGS)

    def test_a1_positive(self):
        """Posterior mean a1 must be positive: clubform_attack predicts initial attack."""
        a1_mean = float(np.mean(self.guide_samples["a1"]))
        assert a1_mean > 0, f"a1 posterior mean={a1_mean:.3f} should be > 0"

    def test_poor_team_anchored(self):
        """Data-poor team's posterior attack must be pulled toward its clubform prior.

        With only 2 matches, the prior dominates: the posterior mean should be
        closer to the clubform-implied value (a0 + a1 * CF_ATT[POOR_TEAM]) than
        to the uninformative alternative of 0.
        """
        a0_mean = float(np.mean(self.guide_samples["a0"]))
        a1_mean = float(np.mean(self.guide_samples["a1"]))
        cf_implied = a0_mean + a1_mean * CF_ATT[POOR_TEAM]

        attack_now = np.array(self.model_samples["attack_now"])  # (N_SAMPLES, N_TEAMS)
        post_att_poor = float(attack_now.mean(axis=0)[POOR_TEAM])

        dist_to_prior = abs(post_att_poor - cf_implied)
        dist_to_zero = abs(post_att_poor)

        assert dist_to_prior < dist_to_zero, (
            f"Poor team not anchored to clubform: "
            f"post_att={post_att_poor:.3f}, cf_implied={cf_implied:.3f}, "
            f"dist_to_prior={dist_to_prior:.3f}, dist_to_zero={dist_to_zero:.3f}"
        )
