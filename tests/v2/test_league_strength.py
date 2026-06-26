"""Tests for cupcast.v2.clubform.league_strength — in-memory fixtures only."""

import pandas as pd

from cupcast.v2.clubform.league_strength import (
    NON_EUROPEAN_LEAGUE_STRENGTH,
    league_strength_index,
    normalize_club_name,
)


def _clubelo_fixture() -> pd.DataFrame:
    """Nine clubs across three leagues, two clubs in a sparse fourth league."""
    return pd.DataFrame(
        {
            "team": [
                "alpha fc", "beta fc", "gamma fc",   # league 1 – strong
                "delta sc", "epsilon sc", "zeta sc",  # league 2 – median
                "eta uc", "theta uc", "iota uc",      # league 3 – weak
                "sparse a", "sparse b",               # league 4 – <3 clubs (fallback)
            ],
            "elo": [
                1800.0, 1850.0, 1900.0,
                1600.0, 1650.0, 1700.0,
                1400.0, 1450.0, 1500.0,
                1550.0, 1600.0,
            ],
            "country": ["ENG"] * 3 + ["ESP"] * 3 + ["ITA"] * 3 + ["FRA"] * 2,
            "level": [1] * 11,
        }
    )


def _club_to_league() -> dict[str, int]:
    return {
        "alpha fc": 1, "beta fc": 1, "gamma fc": 1,
        "delta sc": 2, "epsilon sc": 2, "zeta sc": 2,
        "eta uc": 3, "theta uc": 3, "iota uc": 3,
        "sparse a": 4, "sparse b": 4,
    }


class TestNormalizeClubName:
    def test_accent_strip(self):
        assert normalize_club_name("Atlético Madrid") == "atletico madrid"

    def test_casefold(self):
        assert normalize_club_name("FC Bayern München") == "fc bayern munchen"

    def test_strip_whitespace(self):
        assert normalize_club_name("  Real Madrid  ") == "real madrid"

    def test_already_normalized(self):
        assert normalize_club_name("paris saint-germain") == "paris saint-germain"


class TestLeagueStrengthIndex:
    def test_median_league_returns_one(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert abs(idx[2] - 1.0) < 1e-9, f"expected 1.0, got {idx[2]}"

    def test_strong_league_above_one(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert idx[1] > 1.0, f"expected >1.0, got {idx[1]}"

    def test_weak_league_below_one(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert idx[3] < 1.0, f"expected <1.0, got {idx[3]}"

    def test_sparse_league_falls_back_to_one(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert idx[4] == 1.0, f"expected fallback 1.0, got {idx[4]}"

    def test_multiplier_bounded(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        for lid, val in idx.items():
            assert 0.5 <= val <= 1.5, f"league {lid}: {val} out of [0.5, 1.5]"

    def test_ordering(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert idx[1] > idx[2] > idx[3], "strong > median > weak violated"

    def test_all_leagues_present(self):
        idx = league_strength_index(_clubelo_fixture(), _club_to_league())
        assert set(idx) == {1, 2, 3, 4}

    def test_single_league_returns_one(self):
        """One league → z-score is 0 → multiplier = 1.0."""
        elo = pd.DataFrame(
            {"team": ["club a", "club b", "club c"], "elo": [1600.0, 1650.0, 1700.0],
             "country": ["ENG"] * 3, "level": [1] * 3}
        )
        idx = league_strength_index(elo, {"club a": 99, "club b": 99, "club c": 99})
        assert abs(idx[99] - 1.0) < 1e-9

    def test_unknown_clubs_ignored(self):
        """Clubs in clubelo but absent from club_to_league are skipped."""
        elo = _clubelo_fixture()
        mapping = {"alpha fc": 1, "beta fc": 1, "gamma fc": 1}  # only league 1
        idx = league_strength_index(elo, mapping)
        assert set(idx) == {1}
        assert abs(idx[1] - 1.0) < 1e-9  # one league → z=0 → 1.0


class TestNonEuropeanFallback:
    """NON_EUROPEAN_LEAGUE_STRENGTH is applied to leagues with <3 ClubElo clubs."""

    def _elo_with_sparse_brazil(self) -> pd.DataFrame:
        """ClubElo snapshot: three European clubs + one Brazilian club (sparse)."""
        return pd.DataFrame(
            {
                "team": ["alpha fc", "beta fc", "gamma fc", "flamengo"],
                "elo": [1700.0, 1700.0, 1700.0, 1680.0],
                "country": ["ENG", "ENG", "ENG", "BRA"],
                "level": [1, 1, 1, 1],
            }
        )

    def _elo_with_sparse_mystery(self) -> pd.DataFrame:
        """ClubElo snapshot: three European clubs + one mystery-league club."""
        return pd.DataFrame(
            {
                "team": ["alpha fc", "beta fc", "gamma fc", "mystery club"],
                "elo": [1700.0, 1700.0, 1700.0, 1680.0],
                "country": ["ENG", "ENG", "ENG", "XX"],
                "level": [1, 1, 1, 1],
            }
        )

    def test_curated_league_gets_curated_value(self):
        """Brazil Serie A (71) with <3 ClubElo clubs → 0.90."""
        elo = self._elo_with_sparse_brazil()
        # One Brazil club in ClubElo → sub-threshold → curated fallback.
        mapping = {
            "alpha fc": 39, "beta fc": 39, "gamma fc": 39,
            "flamengo": 71,
        }
        idx = league_strength_index(elo, mapping)
        assert abs(idx[71] - 0.90) < 1e-9, f"expected 0.90, got {idx[71]}"

    def test_unknown_sub_threshold_league_gets_one(self):
        """A league id absent from the curated dict still falls back to 1.0."""
        elo = self._elo_with_sparse_mystery()
        mapping = {
            "alpha fc": 39, "beta fc": 39, "gamma fc": 39,
            "mystery club": 9999,
        }
        idx = league_strength_index(elo, mapping)
        assert idx[9999] == 1.0, f"expected 1.0, got {idx[9999]}"

    def test_well_covered_league_uses_clubelo_not_curated(self):
        """A league with ≥3 ClubElo clubs uses the derived multiplier, not the curated dict."""
        elo = pd.DataFrame(
            {
                "team": ["flamengo", "palmeiras", "corinthians", "alpha fc"],
                "elo": [1800.0, 1820.0, 1840.0, 1200.0],
                "country": ["BRA", "BRA", "BRA", "ENG"],
                "level": [1, 1, 1, 1],
            }
        )
        mapping = {
            "flamengo": 71, "palmeiras": 71, "corinthians": 71,
            "alpha fc": 39,
        }
        idx = league_strength_index(elo, mapping)
        # League 71 has ≥3 matched clubs → ClubElo-derived multiplier, not 0.90.
        assert idx[71] != 0.90, (
            "well-covered league should use ClubElo-derived value, not curated fallback"
        )

    def test_curated_dict_contains_expected_keys(self):
        """Smoke-check the public constant has the five documented leagues."""
        assert set(NON_EUROPEAN_LEAGUE_STRENGTH) == {71, 128, 262, 253, 307}


def test_zero_clubelo_match_league_gets_curated_fallback():
    """Leagues with ZERO ClubElo matches must still receive the curated fallback.

    Regression: non-European leagues (no ClubElo clubs) never reached the index,
    so apply_league_strength fell back to a neutral 1.0 instead of the curated value.
    """
    clubelo = pd.DataFrame(
        {
            "team": ["alpha fc", "beta fc", "gamma fc", "delta fc", "epsilon fc", "zeta fc"],
            "elo": [1800.0, 1850.0, 1900.0, 1500.0, 1550.0, 1600.0],
            "country": ["ENG"] * 3 + ["ESP"] * 3,
            "level": [1] * 6,
        }
    )
    club_to_league = {
        "alpha fc": 39, "beta fc": 39, "gamma fc": 39,
        "delta fc": 140, "epsilon fc": 140, "zeta fc": 140,
        "al hilal": 307, "al nassr": 307,   # Saudi: zero ClubElo matches
        "team x": 999, "team y": 999,        # unknown league: zero matches
    }
    idx = league_strength_index(clubelo, club_to_league)
    assert idx[307] == NON_EUROPEAN_LEAGUE_STRENGTH[307]   # curated fallback applied
    assert idx[999] == 1.0                                 # unknown league -> neutral
    assert idx[39] != 1.0 and idx[140] != 1.0              # two covered leagues -> z-scored
