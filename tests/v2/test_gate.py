"""Tests for the two-source forecast gate orchestrator."""

from __future__ import annotations

import pytest

from cupcast.v2.fetch.espn import canon
from cupcast.v2.gate import (
    api_group_scores,
    api_group_standings,
    build_reports,
    espn_r32_pairs,
    verify_sources,
)
from cupcast.v2.reconcile import ReconcileError
from cupcast.v2.sim.structure import GROUP_LETTERS, GROUPS, R32

_THIRD_GROUPS = ("B", "D", "E", "F", "I", "J", "K", "L")


def _resolve_wr(spec, winners, runners):
    kind, value = spec
    return winners[value] if kind == "W" else runners[value]


def _consistent_world():
    """A fully self-consistent dataset across both sources for all 16 R32 slots."""
    winners = {g: GROUPS[g][0] for g in GROUPS}
    runners = {g: GROUPS[g][1] for g in GROUPS}
    thirds = iter(GROUPS[g][2] for g in _THIRD_GROUPS)

    r32_fixtures, espn_pairs = [], []
    for _match, spec_a, spec_b, _venue in R32:
        ta = _resolve_wr(spec_a, winners, runners)
        tb = next(thirds) if spec_b[0] == "T" else _resolve_wr(spec_b, winners, runners)
        r32_fixtures.append(
            {"teams": {"home": {"name": ta}, "away": {"name": tb}},
             "league": {"round": "Round of 32"}}
        )
        espn_pairs.append({"home": canon(ta), "away": canon(tb)})

    group_fixtures = [
        {"teams": {"home": {"name": GROUPS["A"][0]}, "away": {"name": GROUPS["A"][1]}},
         "goals": {"home": 2, "away": 1}, "league": {"round": "Group Stage - 1"}},
        {"teams": {"home": {"name": GROUPS["B"][0]}, "away": {"name": GROUPS["B"][1]}},
         "goals": {"home": 0, "away": 0}, "league": {"round": "Group Stage - 1"}},
    ]
    espn_scores = [
        {"home": canon(GROUPS["A"][0]), "away": canon(GROUPS["A"][1]),
         "home_goals": 2, "away_goals": 1},
        {"home": canon(GROUPS["B"][0]), "away": canon(GROUPS["B"][1]),
         "home_goals": 0, "away_goals": 0},
    ]

    api_standings = [{"league": {"standings": [
        [{"group": f"Group {g}", "rank": i + 1, "team": {"name": GROUPS[g][i]}}
         for i in range(4)]
        for g in GROUP_LETTERS
    ]}}]
    espn_standings = {
        g: [{"team": canon(GROUPS[g][i]), "rank": i + 1} for i in range(4)]
        for g in GROUP_LETTERS
    }
    espn_fixtures = espn_pairs + [{"home": "roundof321winner", "away": "roundof323winner"}]

    return {
        "fixtures": group_fixtures + r32_fixtures,
        "api_standings": api_standings,
        "espn_scores": espn_scores,
        "espn_standings": espn_standings,
        "espn_fixtures": espn_fixtures,
    }


class _FakeAF:
    def __init__(self, fixtures, standings):
        self._fixtures, self._standings = fixtures, standings

    def get_response(self, endpoint, params):
        if endpoint == "fixtures":
            return self._fixtures
        if endpoint == "standings":
            return self._standings
        raise KeyError(endpoint)


class _FakeESPN:
    def __init__(self, scores, standings, fixtures):
        self._scores, self._standings, self._fixtures = scores, standings, fixtures

    def completed_group_results(self, start, end):
        return self._scores

    def final_standings(self, season):
        return self._standings

    def scheduled_fixtures(self, start, end):
        return self._fixtures


def _clients(world):
    af = _FakeAF(world["fixtures"], world["api_standings"])
    espn = _FakeESPN(world["espn_scores"], world["espn_standings"], world["espn_fixtures"])
    return af, espn


def test_build_reports_all_agree():
    world = _consistent_world()
    reports = build_reports(
        world["fixtures"], world["api_standings"], world["espn_scores"],
        world["espn_standings"], world["espn_fixtures"],
    )
    assert {r.label for r in reports} == {"scores", "standings", "matchups"}
    assert all(r.ok for r in reports)
    matchups = next(r for r in reports if r.label == "matchups")
    assert matchups.matched == 16


def test_verify_sources_returns_reports_when_agree():
    af, espn = _clients(_consistent_world())
    reports = verify_sources(af, espn)
    assert len(reports) == 3
    assert all(r.ok for r in reports)


def test_verify_sources_raises_on_score_mismatch():
    world = _consistent_world()
    world["espn_scores"][0]["home_goals"] = 5
    af, espn = _clients(world)
    with pytest.raises(ReconcileError):
        verify_sources(af, espn)


def test_verify_sources_raises_on_matchup_mismatch():
    world = _consistent_world()
    world["espn_fixtures"][0] = {"home": "narnia", "away": "atlantis"}
    af, espn = _clients(world)
    with pytest.raises(ReconcileError):
        verify_sources(af, espn)


def test_api_group_standings_drops_aggregate_stage_row():
    payload = [{"league": {"standings": [
        [{"group": "Group A", "rank": 1, "team": {"name": "Mexico"}},
         {"group": "Group A", "rank": 2, "team": {"name": "South Africa"}}],
        [{"group": "Group Stage", "rank": 1, "team": {"name": "Mexico"}}],
    ]}}]
    table = api_group_standings(payload)
    assert "Stage" not in table
    assert [r["team"] for r in table["A"]] == ["Mexico", "South Africa"]


def test_espn_r32_pairs_excludes_placeholder_feeders():
    fixtures = [
        {"home": "brazil", "away": "japan"},
        {"home": "roundof321winner", "away": "roundof323winner"},
    ]
    assert espn_r32_pairs(fixtures) == [frozenset({"brazil", "japan"})]


def test_api_group_scores_keeps_only_completed_group_games():
    fixtures = [
        {"teams": {"home": {"name": "Brazil"}, "away": {"name": "Morocco"}},
         "goals": {"home": 1, "away": 0}, "league": {"round": "Group Stage - 2"}},
        {"teams": {"home": {"name": "Brazil"}, "away": {"name": "Japan"}},
         "goals": {"home": None, "away": None}, "league": {"round": "Round of 32"}},
    ]
    scores = api_group_scores(fixtures)
    assert len(scores) == 1
    assert scores[0]["home"] == "Brazil"
