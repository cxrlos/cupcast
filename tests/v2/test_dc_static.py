"""
Synthetic-data recovery test for static_dc_model.

Generates match data from known per-team attack values, fits the NumPyro model
via NUTS, and asserts the posterior mean attack preserves the known ranking.

JAX is forced to CPU to keep wall-time predictable.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import numpyro
from jax import random
from numpyro.infer import MCMC, NUTS

from cupcast.v2.model.dixon_coles import dc_log_tau, static_dc_model, team_index

numpyro.set_host_device_count(1)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

N_TEAMS = 4
# Well-separated attack strengths (highest → lowest)
_ATTACK_TRUE = np.array([0.9, 0.3, -0.3, -0.9], dtype=float)
_DEFENSE_TRUE = np.zeros(N_TEAMS, dtype=float)
_MU_TRUE = 0.0
_RHO_TRUE = 0.0
_GAMMA_TRUE = 0.0

_RNG = np.random.default_rng(2026)

# All ordered team pairs (not self), repeated to reach ~240 matches.
_PAIRS = [(h, a) for h in range(N_TEAMS) for a in range(N_TEAMS) if h != a]
_REPS = 20  # 12 pairs * 20 = 240 matches
_ALL_PAIRS = (_PAIRS * _REPS)

_HOME_IDX = np.array([p[0] for p in _ALL_PAIRS], dtype=np.int32)
_AWAY_IDX = np.array([p[1] for p in _ALL_PAIRS], dtype=np.int32)
_HOST = np.zeros(len(_ALL_PAIRS), dtype=float)

_LAM_TRUE = np.exp(_MU_TRUE + _ATTACK_TRUE[_HOME_IDX] - _DEFENSE_TRUE[_AWAY_IDX])
_NU_TRUE = np.exp(_MU_TRUE + _ATTACK_TRUE[_AWAY_IDX] - _DEFENSE_TRUE[_HOME_IDX])
_X = _RNG.poisson(_LAM_TRUE).astype(np.int32)
_Y = _RNG.poisson(_NU_TRUE).astype(np.int32)


# ---------------------------------------------------------------------------
# dc_log_tau unit tests
# ---------------------------------------------------------------------------


class TestDcLogTau:
    def test_returns_zero_for_large_scores(self):
        import jax.numpy as jnp

        x = jnp.array([3, 4, 5])
        y = jnp.array([2, 3, 4])
        lam = jnp.ones(3)
        nu = jnp.ones(3)
        rho = 0.1
        log_tau = dc_log_tau(x, y, lam, nu, rho)
        # All large-score corrections are log(1) = 0
        np.testing.assert_allclose(np.array(log_tau), 0.0, atol=1e-6)

    def test_00_case_reduces_tau(self):
        import jax.numpy as jnp

        x = jnp.array([0])
        y = jnp.array([0])
        lam = jnp.array([1.0])
        nu = jnp.array([1.0])
        rho = 0.1
        # tau = 1 - 1*1*0.1 = 0.9  → log(0.9)
        log_tau = dc_log_tau(x, y, lam, nu, rho)
        np.testing.assert_allclose(np.array(log_tau), np.log(0.9), atol=1e-6)

    def test_11_case(self):
        import jax.numpy as jnp

        x = jnp.array([1])
        y = jnp.array([1])
        lam = jnp.array([2.0])
        nu = jnp.array([1.5])
        rho = 0.1
        # tau = 1 - 0.1 = 0.9 → log(0.9)
        log_tau = dc_log_tau(x, y, lam, nu, rho)
        np.testing.assert_allclose(np.array(log_tau), np.log(0.9), atol=1e-6)

    def test_01_case(self):
        import jax.numpy as jnp

        x = jnp.array([0])
        y = jnp.array([1])
        lam = jnp.array([2.0])
        nu = jnp.array([1.5])
        rho = 0.1
        # tau = 1 + 2.0*0.1 = 1.2 → log(1.2)
        log_tau = dc_log_tau(x, y, lam, nu, rho)
        np.testing.assert_allclose(np.array(log_tau), np.log(1.2), atol=1e-6)

    def test_10_case(self):
        import jax.numpy as jnp

        x = jnp.array([1])
        y = jnp.array([0])
        lam = jnp.array([2.0])
        nu = jnp.array([1.5])
        rho = 0.1
        # tau = 1 + 1.5*0.1 = 1.15 → log(1.15)
        log_tau = dc_log_tau(x, y, lam, nu, rho)
        np.testing.assert_allclose(np.array(log_tau), np.log(1.15), atol=1e-6)


# ---------------------------------------------------------------------------
# team_index helper test
# ---------------------------------------------------------------------------


class TestTeamIndex:
    def test_contiguous_indices(self):
        import pandas as pd

        df = pd.DataFrame({"home": ["A", "B", "C"], "away": ["B", "C", "A"]})
        teams, h_idx, a_idx = team_index(df)
        n = len(teams)
        assert set(h_idx) | set(a_idx) == set(range(n))

    def test_sorted_team_order(self):
        import pandas as pd

        df = pd.DataFrame({"home": ["Brazil", "Argentina"], "away": ["Argentina", "Brazil"]})
        teams, _, _ = team_index(df)
        assert teams == tuple(sorted(teams))


# ---------------------------------------------------------------------------
# NUTS recovery test
# ---------------------------------------------------------------------------


class TestNutsRecovery:
    def test_attack_ranking_recovered(self):
        """Posterior mean attack must rank teams in the same order as attack_true."""
        mcmc = MCMC(
            NUTS(static_dc_model),
            num_warmup=200,
            num_samples=200,
            progress_bar=False,
        )
        mcmc.run(
            random.PRNGKey(0),
            home_idx=_HOME_IDX,
            away_idx=_AWAY_IDX,
            host=_HOST,
            x=_X,
            y=_Y,
            n_teams=N_TEAMS,
        )
        samples = mcmc.get_samples()
        post_attack_mean = np.array(samples["attack"]).mean(axis=0)
        recovered_ranking = np.argsort(post_attack_mean)[::-1]
        expected_ranking = np.argsort(_ATTACK_TRUE)[::-1]
        assert list(recovered_ranking) == list(expected_ranking), (
            f"Posterior attack ranking {list(recovered_ranking)} "
            f"does not match truth {list(expected_ranking)}. "
            f"Post means: {post_attack_mean}, true: {_ATTACK_TRUE}"
        )
