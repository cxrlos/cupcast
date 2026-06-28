"""Tests for the two-source reconcile gate — pure data, no network."""

from __future__ import annotations

import pytest

from cupcast.v2.reconcile import (
    ReconcileError,
    ReconcileReport,
    assert_sources_agree,
    reconcile_matchups,
    reconcile_scores,
    reconcile_standings,
)


# ---------------------------------------------------------------------------
# reconcile_scores
# ---------------------------------------------------------------------------

def _score(home: str, away: str, hg: int, ag: int) -> dict:
    return {"home": home, "away": away, "home_goals": hg, "away_goals": ag}


class TestReconcileScores:
    def test_all_matched_returns_ok(self):
        api = [_score("Brazil", "Morocco", 2, 0), _score("france", "senegal", 3, 1)]
        espn = [_score("brazil", "morocco", 2, 0), _score("France", "Senegal", 3, 1)]
        report = reconcile_scores(api, espn)
        assert report.ok is True
        assert report.matched == 2
        assert report.mismatches == []
        assert report.missing == []
        assert report.label == "scores"

    def test_score_mismatch_detected(self):
        api = [_score("Brazil", "Morocco", 2, 0)]
        espn = [_score("brazil", "morocco", 1, 0)]  # different score
        report = reconcile_scores(api, espn)
        assert report.ok is False
        assert report.matched == 0
        assert len(report.mismatches) == 1
        m = report.mismatches[0]
        assert "brazil" in m["teams"] and "morocco" in m["teams"]

    def test_game_only_in_api_flagged_missing(self):
        api = [_score("Brazil", "Morocco", 2, 0), _score("France", "Senegal", 1, 0)]
        espn = [_score("brazil", "morocco", 2, 0)]
        report = reconcile_scores(api, espn)
        assert report.ok is False
        assert len(report.missing) == 1
        assert report.missing[0]["source"] == "espn"

    def test_game_only_in_espn_flagged_missing(self):
        api = [_score("Brazil", "Morocco", 2, 0)]
        espn = [_score("brazil", "morocco", 2, 0), _score("France", "Senegal", 1, 0)]
        report = reconcile_scores(api, espn)
        assert report.ok is False
        assert report.missing[0]["source"] == "api"

    def test_canonical_names_match_across_sources(self):
        # ESPN uses "United States", API-Football uses "USA"
        api = [_score("USA", "Türkiye", 1, 0)]
        espn = [_score("United States", "Turkey", 1, 0)]
        report = reconcile_scores(api, espn)
        assert report.ok is True
        assert report.matched == 1

    def test_empty_sources_ok(self):
        assert reconcile_scores([], []).ok is True


# ---------------------------------------------------------------------------
# reconcile_matchups
# ---------------------------------------------------------------------------

class TestReconcileMatchups:
    def test_all_matched_returns_ok(self):
        our = {73: ("Brazil", "Morocco"), 74: ("France", "Senegal")}
        espn_pairs = [frozenset({"brazil", "morocco"}), frozenset({"france", "senegal"})]
        report = reconcile_matchups(our, espn_pairs)
        assert report.ok is True
        assert report.matched == 2

    def test_our_slot_not_in_espn_flagged(self):
        our = {73: ("Brazil", "Morocco"), 74: ("France", "Germany")}
        espn_pairs = [frozenset({"brazil", "morocco"})]
        report = reconcile_matchups(our, espn_pairs)
        assert report.ok is False
        assert len(report.mismatches) == 1
        assert report.mismatches[0]["slot"] == 74

    def test_espn_pair_not_in_our_flagged(self):
        our = {73: ("Brazil", "Morocco")}
        espn_pairs = [
            frozenset({"brazil", "morocco"}),
            frozenset({"france", "senegal"}),  # extra ESPN pair
        ]
        report = reconcile_matchups(our, espn_pairs)
        assert report.ok is False
        assert len(report.missing) == 1

    def test_canon_applied_across_sources(self):
        our = {81: ("USA", "Türkiye")}
        espn_pairs = [frozenset({"usa", "turkiye"})]
        report = reconcile_matchups(our, espn_pairs)
        assert report.ok is True

    def test_empty_both_ok(self):
        assert reconcile_matchups({}, []).ok is True


# ---------------------------------------------------------------------------
# reconcile_standings
# ---------------------------------------------------------------------------

def _row(team: str, rank: int, points: int = 0) -> dict:
    return {"team": team, "rank": rank, "points": points, "note": ""}


class TestReconcileStandings:
    def test_matching_order_ok(self):
        api = {"A": [_row("Mexico", 1), _row("South Korea", 2), _row("South Africa", 3), _row("Czechia", 4)]}
        espn = {"A": [_row("mexico", 1), _row("south korea", 2), _row("south africa", 3), _row("czechia", 4)]}
        report = reconcile_standings(api, espn)
        assert report.ok is True
        assert report.matched == 1

    def test_rank_order_mismatch_detected(self):
        api = {"A": [_row("Mexico", 1), _row("South Korea", 2)]}
        espn = {"A": [_row("South Korea", 1), _row("Mexico", 2)]}
        report = reconcile_standings(api, espn)
        assert report.ok is False
        assert len(report.mismatches) == 1
        assert report.mismatches[0]["group"] == "A"

    def test_group_missing_in_espn_flagged(self):
        api = {"A": [_row("Mexico", 1)], "B": [_row("Canada", 1)]}
        espn = {"A": [_row("Mexico", 1)]}
        report = reconcile_standings(api, espn)
        assert report.ok is False
        assert any(m["source"] == "espn" and m["group"] == "B" for m in report.missing)

    def test_canon_applied_to_names(self):
        api = {"D": [_row("USA", 1), _row("Türkiye", 2)]}
        espn = {"D": [_row("United States", 1), _row("Turkey", 2)]}
        report = reconcile_standings(api, espn)
        assert report.ok is True


# ---------------------------------------------------------------------------
# assert_sources_agree
# ---------------------------------------------------------------------------

def _ok_report(label: str = "test") -> ReconcileReport:
    return ReconcileReport(ok=True, matched=3, mismatches=[], missing=[], label=label)


def _bad_report(label: str = "test") -> ReconcileReport:
    return ReconcileReport(
        ok=False,
        matched=0,
        mismatches=[{"teams": ["a", "b"], "api": (1, 2), "espn": (0, 2)}],
        missing=[],
        label=label,
    )


class TestAssertSourcesAgree:
    def test_all_ok_returns_none(self):
        result = assert_sources_agree(_ok_report("scores"), _ok_report("matchups"))
        assert result is None

    def test_one_failure_raises_reconcile_error(self):
        with pytest.raises(ReconcileError):
            assert_sources_agree(_ok_report("scores"), _bad_report("matchups"))

    def test_error_message_names_failed_label(self):
        with pytest.raises(ReconcileError, match="matchups"):
            assert_sources_agree(_ok_report("scores"), _bad_report("matchups"))

    def test_all_failing_raises_once(self):
        with pytest.raises(ReconcileError):
            assert_sources_agree(_bad_report("scores"), _bad_report("standings"))

    def test_error_lists_mismatch_details(self):
        with pytest.raises(ReconcileError, match="MISMATCH"):
            assert_sources_agree(_bad_report("scores"))
