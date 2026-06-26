"""Regulation + extra-time + GK-shootout knockout sampler."""

from __future__ import annotations

import numpy as np

from cupcast.v2.model.predict import score_matrix
from cupcast.v2.sim.structure import HOST_COUNTRIES

EXTRA_TIME_SCALE = 1.0 / 3.0
# Keeper shot-stopping nudge on the shootout: one standard deviation of
# PSxG+/- per 90 is worth ~2pp, capped at 5pp either way.
GK_SHOOTOUT_BETA = 0.02
GK_SHOOTOUT_CAP = 0.05


def shootout_probability(gk_edge: float) -> float:
    """P(team advances via shootout) given the GK Z-score edge."""
    return 0.5 + float(np.clip(GK_SHOOTOUT_BETA * gk_edge, -GK_SHOOTOUT_CAP, GK_SHOOTOUT_CAP))


def advance_probability(
    posterior,
    team: str,
    opponent: str,
    venue_country: str,
    gk_z: dict[str, float] | None = None,
) -> float:
    """P(team advances) through regulation, extra time, and shootout."""
    host_team = team in HOST_COUNTRIES and team == venue_country
    host_opp = opponent in HOST_COUNTRIES and opponent == venue_country
    regulation = score_matrix(
        posterior, team, opponent, host_home=host_team, host_away=host_opp
    )
    p_win = float(np.tril(regulation, -1).sum())
    p_draw = float(np.trace(regulation))
    extra = score_matrix(
        posterior,
        team,
        opponent,
        host_home=host_team,
        host_away=host_opp,
        rate_scale=EXTRA_TIME_SCALE,
    )
    p_win_et = float(np.tril(extra, -1).sum())
    p_draw_et = float(np.trace(extra))
    gk_z = gk_z or {}
    p_shootout = shootout_probability(gk_z.get(team, 0.0) - gk_z.get(opponent, 0.0))
    return p_win + p_draw * (p_win_et + p_draw_et * p_shootout)


class KnockoutSampler:
    """Samples knockout winners across simulations, caching pair probabilities."""

    def __init__(
        self,
        posterior,
        team_names: tuple[str, ...],
        gk_z: dict[str, float] | None = None,
    ) -> None:
        self.posterior = posterior
        self.team_names = team_names
        self.gk_z = gk_z or {}
        self._cache: dict[tuple[int, int, str], float] = {}

    def _advance_p(self, team_id: int, opp_id: int, venue_country: str) -> float:
        key = (team_id, opp_id, venue_country)
        if key not in self._cache:
            self._cache[key] = advance_probability(
                self.posterior,
                self.team_names[team_id],
                self.team_names[opp_id],
                venue_country,
                self.gk_z,
            )
        return self._cache[key]

    def sample_winners(
        self,
        side_a: np.ndarray,
        side_b: np.ndarray,
        venue_country: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return winner indices (one of side_a or side_b per sim), vectorised."""
        pair_codes = side_a * len(self.team_names) + side_b
        p_advance = np.empty(side_a.shape, dtype=np.float64)
        for code in np.unique(pair_codes):
            a, b = divmod(int(code), len(self.team_names))
            p_advance[pair_codes == code] = self._advance_p(a, b, venue_country)
        a_wins = rng.random(side_a.shape) < p_advance
        return np.where(a_wins, side_a, side_b)
