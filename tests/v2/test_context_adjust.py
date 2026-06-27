"""Tests for context.adjust: bounded, audited prediction adjustments (Task 2 TDD)."""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

from cupcast.v2.context.adjust import (
    CAPS,
    adjusted_outcome_probs,
    covariate_log_effects,
    rate_multipliers,
)
from cupcast.v2.model.fit import Posterior


def _make_posterior() -> Posterior:
    teams = ("A", "B", "C")
    attack = np.array([0.5, 0.0, -0.5])
    defense = np.array([-0.3, 0.0, 0.3])
    return Posterior(teams=teams, attack=attack, defense=defense, mu=0.3, gamma=0.2, rho=-0.05)


def _zero_ctx() -> dict:
    return {"is_host": False, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}


# ---------------------------------------------------------------------------
# covariate_log_effects
# ---------------------------------------------------------------------------


class TestCovariateLogEffects:
    def test_extreme_travel_capped(self):
        ctx = {"is_host": True, "travel_km": 100_000.0, "rest_days": 0, "altitude_m": 5000}
        assert abs(covariate_log_effects(ctx)["travel"]) <= CAPS["travel"] + 1e-10

    def test_extreme_rest_capped(self):
        ctx = {"is_host": True, "travel_km": 100_000.0, "rest_days": 0, "altitude_m": 5000}
        assert abs(covariate_log_effects(ctx)["rest"]) <= CAPS["rest"] + 1e-10

    def test_extreme_altitude_non_host_capped(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 4, "altitude_m": 50_000}
        assert abs(covariate_log_effects(ctx)["altitude"]) <= CAPS["altitude"] + 1e-10

    def test_host_at_altitude_no_altitude_penalty(self):
        """Host at 2240 m gets +host, not an altitude penalty."""
        ctx = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 2240}
        effects = covariate_log_effects(ctx)
        assert effects["altitude"] == pytest.approx(0.0)
        assert effects["host"] == pytest.approx(CAPS["host"])

    def test_non_host_at_altitude_gets_penalty(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 4, "altitude_m": 2240}
        assert covariate_log_effects(ctx)["altitude"] < 0.0

    def test_zero_context_all_effects_zero(self):
        effects = covariate_log_effects(_zero_ctx())
        for key, val in effects.items():
            assert val == pytest.approx(0.0), f"{key} should be 0 but got {val}"

    def test_host_effect_equals_cap(self):
        ctx = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        assert covariate_log_effects(ctx)["host"] == pytest.approx(CAPS["host"])

    def test_no_host_effect_zero(self):
        assert covariate_log_effects(_zero_ctx())["host"] == pytest.approx(0.0)

    def test_rest_days_none_no_penalty(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": None, "altitude_m": 500}
        assert covariate_log_effects(ctx)["rest"] == pytest.approx(0.0)

    def test_rest_days_gte_4_no_penalty(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 5, "altitude_m": 500}
        assert covariate_log_effects(ctx)["rest"] == pytest.approx(0.0)

    def test_altitude_below_1000_no_penalty(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 4, "altitude_m": 800}
        assert covariate_log_effects(ctx)["altitude"] == pytest.approx(0.0)

    def test_altitude_none_no_penalty(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 4, "altitude_m": None}
        assert covariate_log_effects(ctx)["altitude"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# rate_multipliers
# ---------------------------------------------------------------------------


class TestRateMultipliers:
    def test_zero_context_multipliers_are_one(self):
        mh, ma, _ = rate_multipliers(_zero_ctx(), _zero_ctx())
        assert mh == pytest.approx(1.0)
        assert ma == pytest.approx(1.0)

    def test_zero_context_empty_provenance(self):
        _, _, prov = rate_multipliers(_zero_ctx(), _zero_ctx())
        assert prov == []

    def test_provenance_records_nonzero_home_covariates(self):
        ctx_home = {"is_host": True, "travel_km": 2000.0, "rest_days": 2, "altitude_m": 500}
        _, _, prov = rate_multipliers(ctx_home, _zero_ctx())
        home_covariates = {r["covariate"] for r in prov if r["side"] == "home"}
        assert "host" in home_covariates
        assert "travel" in home_covariates
        assert "rest" in home_covariates

    def test_provenance_record_has_required_keys(self):
        ctx_home = {"is_host": False, "travel_km": 1500.0, "rest_days": 4, "altitude_m": 500}
        _, _, prov = rate_multipliers(ctx_home, _zero_ctx())
        rec = next(r for r in prov if r["covariate"] == "travel" and r["side"] == "home")
        assert rec["raw"] == pytest.approx(1500.0)
        assert "log_effect" in rec

    def test_provenance_away_side_labeled(self):
        ctx_away = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        _, _, prov = rate_multipliers(_zero_ctx(), ctx_away)
        assert any(r["side"] == "away" for r in prov)

    def test_host_context_multiplier_gt_one(self):
        ctx = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        mh, _, _ = rate_multipliers(ctx, _zero_ctx())
        assert mh > 1.0

    def test_long_travel_multiplier_lt_one(self):
        ctx = {"is_host": False, "travel_km": 3000.0, "rest_days": 4, "altitude_m": 500}
        mh, _, _ = rate_multipliers(ctx, _zero_ctx())
        assert mh < 1.0

    def test_short_rest_multiplier_lt_one(self):
        ctx = {"is_host": False, "travel_km": 0.0, "rest_days": 2, "altitude_m": 500}
        mh, _, _ = rate_multipliers(ctx, _zero_ctx())
        assert mh < 1.0


# ---------------------------------------------------------------------------
# adjusted_outcome_probs
# ---------------------------------------------------------------------------


class TestAdjustedOutcomeProbs:
    def test_zero_context_equals_base_outcome_probs(self):
        """Critical ablation-baseline: zero context must exactly reproduce base outcome_probs."""
        from cupcast.v2.model.predict import outcome_probs

        post = _make_posterior()
        base = outcome_probs(post, "A", "C", host_home=False, host_away=False)
        (ph, pd_, pa), _ = adjusted_outcome_probs(post, "A", "C", _zero_ctx(), _zero_ctx())
        assert (ph, pd_, pa) == pytest.approx(base, abs=1e-12)

    def test_probs_sum_to_one(self):
        post = _make_posterior()
        ctx = {"is_host": True, "travel_km": 500.0, "rest_days": 3, "altitude_m": 500}
        (ph, pd_, pa), _ = adjusted_outcome_probs(post, "A", "C", ctx, _zero_ctx())
        assert abs(ph + pd_ + pa - 1.0) < 1e-10

    def test_host_home_raises_win_prob(self):
        post = _make_posterior()
        (ph_base, _, _), _ = adjusted_outcome_probs(post, "A", "C", _zero_ctx(), _zero_ctx())
        ctx_host = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        (ph_host, _, _), _ = adjusted_outcome_probs(post, "A", "C", ctx_host, _zero_ctx())
        assert ph_host > ph_base

    def test_host_home_bounded(self):
        """Win-prob lift from host effect must stay below an absolute 0.15 margin."""
        post = _make_posterior()
        (ph_base, _, _), _ = adjusted_outcome_probs(post, "A", "C", _zero_ctx(), _zero_ctx())
        ctx_host = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        (ph_host, _, _), _ = adjusted_outcome_probs(post, "A", "C", ctx_host, _zero_ctx())
        assert (ph_host - ph_base) < 0.15

    def test_provenance_is_list_of_dicts(self):
        post = _make_posterior()
        ctx = {"is_host": True, "travel_km": 0.0, "rest_days": 4, "altitude_m": 500}
        _, prov = adjusted_outcome_probs(post, "A", "C", ctx, _zero_ctx())
        assert isinstance(prov, list)
        assert all(isinstance(r, dict) for r in prov)

    def test_zero_context_empty_provenance(self):
        post = _make_posterior()
        _, prov = adjusted_outcome_probs(post, "A", "C", _zero_ctx(), _zero_ctx())
        assert prov == []

    def test_host_away_flag_threaded(self):
        """host_away=True flag must be passed through to score_matrix."""
        from cupcast.v2.model.predict import outcome_probs

        post = _make_posterior()
        (_, _, pa_adj), _ = adjusted_outcome_probs(
            post, "A", "C", _zero_ctx(), _zero_ctx(), host_home=False, host_away=True
        )
        _, _, pa_base = outcome_probs(post, "A", "C", host_home=False, host_away=True)
        assert pa_adj == pytest.approx(pa_base, abs=1e-12)
