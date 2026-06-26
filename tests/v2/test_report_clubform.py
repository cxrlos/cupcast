"""Tests for cupcast.v2.clubform.report — in-memory fixtures only.

The ``coverage_report`` pure function is fully unit-tested here.
The ``main()`` entry is a real-data smoke and is not unit-tested.
"""

from __future__ import annotations

import pandas as pd

from cupcast.v2.clubform.report import coverage_report


def _profiles_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["Brazil", "France", "USA", "Morocco", "Japan"],
            "attack": [0.8, 1.2, -0.3, 0.1, 0.5],
            "defense": [0.6, 0.9, -0.1, 0.4, 0.3],
            "gk": [0.2, 0.7, -0.2, 0.1, 0.0],
            "coverage": [0.95, 0.88, 0.72, 0.60, 0.80],
            "n_players": [18, 20, 15, 14, 17],
        }
    )


class TestCoverageReport:
    def test_sorted_by_coverage_descending(self):
        report = coverage_report(_profiles_fixture())
        coverages = report["coverage"].tolist()
        assert coverages == sorted(coverages, reverse=True)

    def test_all_teams_returned_without_filter(self):
        report = coverage_report(_profiles_fixture())
        assert len(report) == 5
        assert set(report["team"]) == {"Brazil", "France", "USA", "Morocco", "Japan"}

    def test_wc_teams_filter(self):
        report = coverage_report(_profiles_fixture(), wc_teams=["Brazil", "France"])
        assert set(report["team"]) == {"Brazil", "France"}

    def test_wc_teams_filter_preserves_sort_order(self):
        report = coverage_report(
            _profiles_fixture(), wc_teams=["Brazil", "France", "Japan"]
        )
        coverages = report["coverage"].tolist()
        assert coverages == sorted(coverages, reverse=True)

    def test_wc_teams_no_match_returns_empty(self):
        report = coverage_report(_profiles_fixture(), wc_teams=["Germany", "Spain"])
        assert report.empty

    def test_output_columns(self):
        report = coverage_report(_profiles_fixture())
        assert list(report.columns) == ["team", "attack", "defense", "gk", "coverage", "n_players"]

    def test_index_is_reset(self):
        report = coverage_report(_profiles_fixture())
        assert list(report.index) == list(range(len(report)))

    def test_original_not_mutated(self):
        profiles = _profiles_fixture()
        original_order = profiles["team"].tolist()
        coverage_report(profiles)
        assert profiles["team"].tolist() == original_order

    def test_single_team(self):
        profiles = pd.DataFrame(
            {
                "team": ["Brazil"],
                "attack": [0.5],
                "defense": [0.3],
                "gk": [0.1],
                "coverage": [0.90],
                "n_players": [20],
            }
        )
        report = coverage_report(profiles)
        assert len(report) == 1
        assert report.iloc[0]["team"] == "Brazil"

    def test_wc_teams_none_returns_all(self):
        report = coverage_report(_profiles_fixture(), wc_teams=None)
        assert len(report) == 5
