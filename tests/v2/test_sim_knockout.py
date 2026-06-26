"""TDD tests for cupcast.v2.sim.knockout."""

from __future__ import annotations

import numpy as np
import pytest

from cupcast.v2.sim.knockout import (
    GK_SHOOTOUT_BETA,
    GK_SHOOTOUT_CAP,
    KnockoutSampler,
    advance_probability,
    shootout_probability,
)

_BASE_RATES: dict[tuple[str, str], tuple[float, float]] = {
    ("Strong", "Weak"): (4.0, 0.1),
    ("Weak", "Strong"): (0.1, 4.0),
    ("Mexico", "Germany"): (1.5, 1.5),
    ("Germany", "Mexico"): (1.5, 1.5),
}

_HOST_BOOST = 1.5


class StubPosterior:
    """Duck-typed stub; only .rate() and .rho needed by predict.score_matrix."""

    rho = 0.0

    def rate(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
    ) -> tuple[float, float]:
        lam, nu = _BASE_RATES.get((home, away), (1.5, 1.5))
        if host_home:
            lam *= _HOST_BOOST
        if host_away:
            nu *= _HOST_BOOST
        return lam, nu


_POSTERIOR = StubPosterior()


def test_shootout_probability_at_zero_edge() -> None:
    assert shootout_probability(0.0) == pytest.approx(0.5)


def test_shootout_probability_clips_positive() -> None:
    assert shootout_probability(1000.0) == pytest.approx(0.5 + GK_SHOOTOUT_CAP)


def test_shootout_probability_clips_negative() -> None:
    assert shootout_probability(-1000.0) == pytest.approx(0.5 - GK_SHOOTOUT_CAP)


def test_shootout_probability_linear_in_range() -> None:
    """Within cap, probability scales linearly with GK edge."""
    edge = 1.0  # well within cap (GK_SHOOTOUT_BETA * 1.0 = 0.02 < 0.05)
    expected = 0.5 + GK_SHOOTOUT_BETA * edge
    assert shootout_probability(edge) == pytest.approx(expected)


def test_advance_probability_strong_beats_weak() -> None:
    p = advance_probability(_POSTERIOR, "Strong", "Weak", "Brazil")
    assert p > 0.5, f"Expected strong team advance probability > 0.5, got {p}"


def test_advance_probability_host_boost() -> None:
    """Mexico at its own venue gets a higher advance probability than at a neutral venue."""
    p_home = advance_probability(_POSTERIOR, "Mexico", "Germany", "Mexico")
    p_neutral = advance_probability(_POSTERIOR, "Mexico", "Germany", "Brazil")
    assert p_home > p_neutral, (
        f"Host advantage expected: {p_home:.4f} > {p_neutral:.4f}"
    )


def test_advance_probability_opponent_host_hurts() -> None:
    """When opponent is the host, the team's advance probability falls below neutral."""
    p_opp_host = advance_probability(_POSTERIOR, "Mexico", "USA", "USA")
    p_neutral = advance_probability(_POSTERIOR, "Mexico", "USA", "Brazil")
    assert p_opp_host < p_neutral, (
        f"Expected lower p when opponent is host: {p_opp_host:.4f} vs neutral {p_neutral:.4f}"
    )


def test_advance_probability_gk_edge_applies() -> None:
    """Positive GK edge for the team raises its advance probability."""
    p_no_gk = advance_probability(_POSTERIOR, "Mexico", "Germany", "Brazil")
    p_with_gk = advance_probability(
        _POSTERIOR, "Mexico", "Germany", "Brazil",
        gk_z={"Mexico": 5.0, "Germany": 0.0},
    )
    assert p_with_gk > p_no_gk, (
        f"GK edge should raise advance prob: {p_with_gk:.4f} vs {p_no_gk:.4f}"
    )


def test_knockout_sampler_returns_valid_sides() -> None:
    """sample_winners must return one of side_a or side_b for each sim."""
    team_names = ("Strong", "Weak")
    sampler = KnockoutSampler(_POSTERIOR, team_names)
    rng = np.random.default_rng(0)
    n = 500
    side_a = np.zeros(n, dtype=np.intp)
    side_b = np.ones(n, dtype=np.intp)
    winners = sampler.sample_winners(side_a, side_b, "Brazil", rng)
    assert winners.shape == (n,)
    assert np.all((winners == 0) | (winners == 1)), "Winners must be indices from side_a or side_b."


def test_knockout_sampler_stronger_wins_more() -> None:
    """With seeded RNG, the stronger team (Strong) wins the large majority of matchups."""
    team_names = ("Strong", "Weak")
    sampler = KnockoutSampler(_POSTERIOR, team_names)
    rng = np.random.default_rng(42)
    n = 1000
    side_a = np.zeros(n, dtype=np.intp)
    side_b = np.ones(n, dtype=np.intp)
    winners = sampler.sample_winners(side_a, side_b, "Brazil", rng)
    strong_wins = int(np.sum(winners == 0))
    assert strong_wins > n * 0.8, (
        f"Expected Strong to win >80% of matchups, won {strong_wins}/{n}"
    )


def test_knockout_sampler_cache_reuse() -> None:
    """Calling sample_winners twice with the same pair uses the cache."""
    team_names = ("Mexico", "Germany")
    sampler = KnockoutSampler(_POSTERIOR, team_names)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    n = 200
    side_a = np.zeros(n, dtype=np.intp)
    side_b = np.ones(n, dtype=np.intp)
    sampler.sample_winners(side_a, side_b, "Brazil", rng1)
    cache_size_first = len(sampler._cache)
    sampler.sample_winners(side_a, side_b, "Brazil", rng2)
    assert len(sampler._cache) == cache_size_first, "Second call must not grow the cache."
