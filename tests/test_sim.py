from itertools import combinations

import numpy as np
import pytest

from cupcast.model.dixon_coles import DixonColesFit
from cupcast.sim.bracket import SLOT_ALLOWED, allocate_thirds
from cupcast.sim.group_stage import PAIRS, group_tables, rank_group
from cupcast.sim.knockout import advance_probability
from cupcast.sim.monte_carlo import run_tournament
from cupcast.sim.worldcup2026 import ALL_TEAMS, GROUPS, TEAM_GROUP


def scores_for(results: dict[tuple[int, int], tuple[int, int]]):
    home = np.zeros((6, 1), dtype=np.int64)
    away = np.zeros((6, 1), dtype=np.int64)
    for m, pair in enumerate(PAIRS):
        home[m, 0], away[m, 0] = results[pair]
    return home, away


def test_group_points_and_goal_difference():
    # Slot 0 wins all three; slot 3 loses all three.
    home, away = scores_for(
        {
            (0, 1): (2, 0),
            (0, 2): (1, 0),
            (0, 3): (3, 1),
            (1, 2): (1, 1),
            (1, 3): (2, 0),
            (2, 3): (1, 0),
        }
    )
    points, gd, gf, ga = group_tables(home, away)
    assert points[:, 0].tolist() == [9, 4, 4, 0]
    assert gd[:, 0].tolist() == [5, 0, 0, -5]
    assert gf[:, 0].tolist() == [6, 3, 2, 1]
    assert ga[:, 0].tolist() == [1, 3, 2, 6]


def test_head_to_head_breaks_full_ties():
    # Slots 0 and 1 finish level on points, GD and GF; 0 beat 1 directly.
    # Slots 2 and 3 also finish level; 3 beat 2 directly.
    results = {
        (0, 1): (1, 0),
        (0, 2): (0, 1),
        (0, 3): (2, 0),
        (1, 2): (2, 0),
        (1, 3): (1, 0),
        (2, 3): (0, 1),
    }
    home, away = scores_for(results)
    rng = np.random.default_rng(0)
    order, *_ = rank_group(home, away, rng)
    assert order[:, 0].tolist() == [0, 1, 3, 2]


def test_every_third_place_combination_allocates():
    for combo in combinations("ABCDEFGHIJKL", 8):
        allocation = allocate_thirds(frozenset(combo))
        assert sorted(allocation.values()) == sorted(combo)
        for slot, group in allocation.items():
            assert group in SLOT_ALLOWED[slot]


def test_allocation_matches_published_annex_example():
    # FIFA's worked example: thirds of groups E-L qualify.
    allocation = allocate_thirds(frozenset("EFGHIJKL"))
    assert allocation == {79: "E", 85: "J", 81: "I", 74: "F", 82: "H", 77: "G", 87: "L", 80: "K"}


def wc_fit(seed=5, spread=0.4):
    rng = np.random.default_rng(seed)
    n = len(ALL_TEAMS)
    attack = rng.normal(0, spread, n)
    defense = rng.normal(0, spread, n)
    return DixonColesFit(
        teams=ALL_TEAMS,
        mu=0.1,
        host_advantage=0.25,
        rho=-0.05,
        attack=attack - attack.mean(),
        defense=defense - defense.mean(),
    )


def test_advance_probability_is_complementary():
    fit = wc_fit()
    p_spain = advance_probability(fit, "Spain", "Uruguay", "United States")
    p_uruguay = advance_probability(fit, "Uruguay", "Spain", "United States")
    assert p_spain + p_uruguay == pytest.approx(1.0, abs=1e-9)
    assert advance_probability(fit, "Mexico", "Japan", "Mexico") > advance_probability(
        fit, "Mexico", "Japan", "United States"
    )


def test_tournament_runs_and_probabilities_are_consistent():
    fit = wc_fit()
    table = run_tournament(fit, n_sims=2000, seed=99)
    assert len(table) == 48
    assert table["p_champion"].sum() == pytest.approx(1.0)
    assert table["p_runner_up"].sum() == pytest.approx(1.0)
    assert table["p_podium_third"].sum() == pytest.approx(1.0)
    assert table["p_final"].sum() == pytest.approx(2.0)
    assert table["p_qualify"].sum() == pytest.approx(32.0)
    by_group = table.groupby("group")["p_group_win"].sum()
    assert np.allclose(by_group, 1.0)
    assert (table["p_qualify"] >= table["p_r16"] - 1e-12).all()
    assert (table["p_r16"] >= table["p_qf"] - 1e-12).all()
    assert (table["p_qf"] >= table["p_sf"] - 1e-12).all()
    assert TEAM_GROUP["Spain"] == "H"
    assert len({team for group in GROUPS.values() for team in group}) == 48


def test_tournament_is_reproducible():
    fit = wc_fit()
    first = run_tournament(fit, n_sims=500, seed=7)
    second = run_tournament(fit, n_sims=500, seed=7)
    assert first.equals(second)
