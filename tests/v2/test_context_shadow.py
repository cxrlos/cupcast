"""TDD tests for morale shadow stub and context_held_out ablation (Task 3)."""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from cupcast.v2.context.ablation import context_held_out
from cupcast.v2.context.morale import morale_signal
from cupcast.v2.model.fit import Posterior

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_posterior() -> Posterior:
    """Four-team posterior covering the fixture set below."""
    teams = ("Mexico", "Brazil", "Argentina", "Germany")
    attack = np.array([0.3, 0.1, 0.0, -0.1])
    defense = np.array([-0.1, -0.1, 0.0, 0.1])
    return Posterior(
        teams=teams, attack=attack, defense=defense, mu=0.2, gamma=0.15, rho=-0.03
    )


def _make_fixtures() -> list[dict]:
    """Three group-stage fixtures covering the ablation test scenarios.

    Fixture 0: Mexico vs Brazil at Mexico City — Mexico is host, altitude 2240 m.
    Fixture 1: Argentina vs Germany at Houston — both teams' first WC match,
               low altitude (15 m), neither is a host → all context effects zero.
    Fixture 2: Brazil vs Unknown at Houston — skipped; "Unknown" not in posterior.
    """
    return [
        {
            "fixture": {
                "date": "2026-06-12T18:00:00+00:00",
                "status": {"short": "FT"},
                "venue": {"city": "Mexico City"},
            },
            "goals": {"home": 2, "away": 1},
            "teams": {"home": {"name": "Mexico"}, "away": {"name": "Brazil"}},
            "league": {"round": "Group Stage - 1"},
        },
        {
            "fixture": {
                "date": "2026-06-12T21:00:00+00:00",
                "status": {"short": "FT"},
                "venue": {"city": "Houston"},
            },
            "goals": {"home": 1, "away": 1},
            "teams": {"home": {"name": "Argentina"}, "away": {"name": "Germany"}},
            "league": {"round": "Group Stage - 1"},
        },
        {
            "fixture": {
                "date": "2026-06-18T18:00:00+00:00",
                "status": {"short": "FT"},
                "venue": {"city": "Houston"},
            },
            "goals": {"home": 0, "away": 1},
            "teams": {"home": {"name": "Brazil"}, "away": {"name": "Unknown"}},
            "league": {"round": "Group Stage - 2"},
        },
    ]


# ---------------------------------------------------------------------------
# morale_signal — shadow assertions
# ---------------------------------------------------------------------------


class TestMoraleSignalShadow:
    def test_weight_is_zero(self):
        assert morale_signal("Mexico")["weight"] == 0.0

    def test_value_is_zero(self):
        assert morale_signal("Mexico")["value"] == 0.0

    def test_sources_empty(self):
        assert morale_signal("Mexico")["sources"] == []

    def test_returns_correct_team(self):
        assert morale_signal("Brazil")["team"] == "Brazil"

    def test_arbitrary_team_name_shadow(self):
        """Shadow weight holds for any team, including unknown ones."""
        assert morale_signal("NonExistentTeam")["weight"] == 0.0

    def test_as_of_ignored(self):
        """as_of parameter is accepted and never changes the weight."""
        from datetime import date

        assert morale_signal("Germany", as_of=date(2026, 6, 15))["weight"] == 0.0


# ---------------------------------------------------------------------------
# context_held_out — shape and alignment
# ---------------------------------------------------------------------------


class TestContextHeldOutShape:
    def test_returns_three_arrays(self):
        post = _make_posterior()
        result = context_held_out(post, _make_fixtures())
        assert len(result) == 3

    def test_k_equals_two_known_finished(self):
        """Fixture 2 is skipped (Unknown not in posterior); k == 2."""
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, _make_fixtures())
        assert len(P_base) == 2
        assert len(P_ctx) == 2
        assert len(y) == 2

    def test_p_base_shape(self):
        post = _make_posterior()
        P_base, _, _ = context_held_out(post, _make_fixtures())
        assert P_base.shape == (2, 3)

    def test_p_ctx_shape(self):
        post = _make_posterior()
        _, P_ctx, _ = context_held_out(post, _make_fixtures())
        assert P_ctx.shape == (2, 3)

    def test_y_shape(self):
        post = _make_posterior()
        _, _, y = context_held_out(post, _make_fixtures())
        assert y.shape == (2,)

    def test_p_base_rows_sum_to_one(self):
        post = _make_posterior()
        P_base, _, _ = context_held_out(post, _make_fixtures())
        np.testing.assert_allclose(P_base.sum(axis=1), np.ones(2), atol=1e-10)

    def test_p_ctx_rows_sum_to_one(self):
        post = _make_posterior()
        _, P_ctx, _ = context_held_out(post, _make_fixtures())
        np.testing.assert_allclose(P_ctx.sum(axis=1), np.ones(2), atol=1e-10)

    def test_y_dtype_int(self):
        post = _make_posterior()
        _, _, y = context_held_out(post, _make_fixtures())
        assert np.issubdtype(y.dtype, np.integer)

    def test_y_values_in_range(self):
        post = _make_posterior()
        _, _, y = context_held_out(post, _make_fixtures())
        assert set(y).issubset({0, 1, 2})

    def test_empty_fixtures_returns_empty_arrays(self):
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, [])
        assert P_base.shape == (0, 3)
        assert P_ctx.shape == (0, 3)
        assert y.shape == (0,)


# ---------------------------------------------------------------------------
# context_held_out — ablation correctness
# ---------------------------------------------------------------------------


class TestContextHeldOutAblation:
    def test_unknown_team_fixture_skipped(self):
        """Fixture 2 has 'Unknown' away; only 2 rows returned, not 3."""
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, _make_fixtures())
        assert P_base.shape[0] == 2

    def test_zero_effects_ctx_equals_base(self):
        """Argentina vs Germany at Houston — all covariates zero → ctx == base."""
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, _make_fixtures())
        # Row 1 is Argentina vs Germany (zero-effect fixture).
        np.testing.assert_allclose(P_ctx[1], P_base[1], atol=1e-12)

    def test_host_context_raises_home_win_prob(self):
        """Mexico at Mexico City — is_host shifts Mexico's win prob UP vs base."""
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, _make_fixtures())
        # Row 0 is Mexico vs Brazil (host fixture); column 0 = home win prob.
        assert P_ctx[0, 0] > P_base[0, 0]

    def test_host_context_shift_bounded(self):
        """Win-prob lift from host context must stay below 0.20."""
        post = _make_posterior()
        P_base, P_ctx, y = context_held_out(post, _make_fixtures())
        assert (P_ctx[0, 0] - P_base[0, 0]) < 0.20

    def test_realized_outcome_row0_is_home_win(self):
        """Mexico 2-1 Brazil → outcome 0 (home win)."""
        post = _make_posterior()
        _, _, y = context_held_out(post, _make_fixtures())
        assert y[0] == 0

    def test_realized_outcome_row1_is_draw(self):
        """Argentina 1-1 Germany → outcome 1 (draw)."""
        post = _make_posterior()
        _, _, y = context_held_out(post, _make_fixtures())
        assert y[1] == 1

    def test_no_host_flags_in_base_call(self):
        """Base and context probs differ only via context for Mexico fixture.

        If host flags were passed to the posterior, the gap would be larger than
        what the context-layer cap alone can produce.  The lift is bounded by
        exp(CAPS['host']) - 1 at the rate level, which translates to < 0.15 on
        the outcome probability for any balanced match-up.
        """
        from cupcast.v2.context.adjust import CAPS

        post = _make_posterior()
        P_base, P_ctx, _ = context_held_out(post, _make_fixtures())
        max_rate_lift = np.exp(CAPS["host"]) - 1  # ≈ 0.127
        assert (P_ctx[0, 0] - P_base[0, 0]) < max_rate_lift + 0.05
