"""TDD tests for cupcast.v2.sim.group_stage."""

from __future__ import annotations

import numpy as np

from cupcast.v2.sim.group_stage import (
    composite_key,
    rank_group,
    sample_group_scores,
)


class StubPosterior:
    """Duck-typed posterior stub; only `.rate()` and `.rho` are needed by predict."""

    rho = 0.0

    def __init__(self, rates: dict[tuple[str, str], tuple[float, float]]) -> None:
        self._rates = rates

    def rate(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
    ) -> tuple[float, float]:
        return self._rates[(home, away)]


_TEAMS = ("A", "B", "C", "D")
_HOSTS_NONE = (False, False, False, False)

# Stub where team A dominates every match (high lam, near-zero nu).
_DOMINANT_RATES: dict[tuple[str, str], tuple[float, float]] = {
    ("A", "B"): (5.0, 0.01),
    ("A", "C"): (5.0, 0.01),
    ("A", "D"): (5.0, 0.01),
    ("B", "C"): (1.5, 1.0),
    ("B", "D"): (1.5, 1.0),
    ("C", "D"): (1.5, 1.0),
}


def test_rank_group_dominant_team_always_first() -> None:
    """Team A's extremal rates make it win all three matches in every sim."""
    posterior = StubPosterior(_DOMINANT_RATES)
    rng = np.random.default_rng(42)
    n_sims = 200
    hg, ag = sample_group_scores(posterior, _TEAMS, _HOSTS_NONE, n_sims, rng)
    order, _, _, _ = rank_group(hg, ag, rng)
    assert np.all(order[0] == 0), "Team A (index 0) must be ranked first in every sim."


def test_composite_key_point_precedence() -> None:
    """More points always outranks fewer points, regardless of GD and GF."""
    # 9 pts, GD=-60, GF=0  vs  6 pts, GD=+100, GF=60
    pts_high = np.array([9, 6])
    gd = np.array([-60, 100])
    gf = np.array([0, 60])
    keys = composite_key(pts_high, gd, gf)
    assert keys[0] > keys[1], "9-point team must outrank a 6-point team in composite key."


def test_composite_key_gd_then_gf_tiebreak() -> None:
    """Within equal points, higher GD wins; within equal pts and GD, higher GF wins."""
    pts = np.array([6, 6, 6])
    gd = np.array([3, 3, 2])
    gf = np.array([5, 4, 9])
    keys = composite_key(pts, gd, gf)
    # Team 0 vs team 1: same pts and GD, team 0 has more GF.
    assert keys[0] > keys[1], "Higher GF breaks equal-pts/equal-GD tie."
    # Team 1 vs team 2: same pts, team 1 has higher GD despite lower GF.
    assert keys[1] > keys[2], "Higher GD breaks equal-pts tie regardless of GF."


def test_rank_group_gd_breaks_points_tie() -> None:
    """Hand-crafted goals array: teams 1 and 2 both have 7 pts, team 1 wins by GD."""
    # PAIRS = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    # Scenario (1 sim):
    #   0 vs 1: 0-1  → team1 3pts GD+1; team0 0pts GD-1
    #   0 vs 2: 0-1  → team2 3pts GD+1; team0 0pts GD-1
    #   0 vs 3: 3-0  → team0 3pts GD+3; team3 0pts GD-3
    #   1 vs 2: 0-0  → team1 1pt;       team2 1pt
    #   1 vs 3: 3-0  → team1 3pts GD+3; team3 0pts GD-3
    #   2 vs 3: 1-0  → team2 3pts GD+1; team3 0pts GD-1
    #
    # Totals:
    #   team0: 3 pts, GF=3, GA=2, GD=+1  → ranks 3rd
    #   team1: 7 pts, GF=4, GA=0, GD=+4  → ranks 1st (more GD than team2)
    #   team2: 7 pts, GF=2, GA=0, GD=+2  → ranks 2nd (less GD than team1)
    #   team3: 0 pts, GF=0, GA=7, GD=-7  → ranks 4th
    home_goals = np.array([[0], [0], [3], [0], [3], [1]], dtype=np.int64)
    away_goals = np.array([[1], [1], [0], [0], [0], [0]], dtype=np.int64)

    rng = np.random.default_rng(0)
    order, pts, gd, gf = rank_group(home_goals, away_goals, rng)

    assert list(order[:, 0]) == [1, 2, 0, 3], (
        f"Expected rank [1,2,0,3], got {list(order[:, 0])}; "
        f"pts={pts[:,0]}, gd={gd[:,0]}, gf={gf[:,0]}"
    )
