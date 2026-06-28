"""Tests for cupcast.v2.clubform.shootout — in-memory data, no network."""

from __future__ import annotations

import pandas as pd
import pytest

from cupcast.v2.clubform.shootout import parse_penalty_stats, team_shootout_z

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pen_player(
    player_id: int,
    name: str,
    position: str,
    minutes: int,
    *,
    pen_scored: int | None = None,
    pen_missed: int | None = None,
    pen_saved: int | None = None,
) -> dict:
    return {
        "player": {"id": player_id, "name": name},
        "statistics": [
            {
                "games": {"minutes": minutes, "position": position, "rating": "7.00"},
                "penalty": {
                    "won": None,
                    "committed": None,
                    "scored": pen_scored,
                    "missed": pen_missed,
                    "saved": pen_saved,
                },
            }
        ],
    }


def _two_team_responses() -> list[dict]:
    """Team A: strong keeper (pen_saved=5) + good takers (80% conversion).
    Team B: weak keeper (pen_saved=0) + poor takers (20% conversion).
    """
    return [
        _pen_player(1, "GK_A", "Goalkeeper", 900, pen_saved=5, pen_scored=0, pen_missed=0),
        _pen_player(2, "FWD_A", "Attacker", 900, pen_scored=4, pen_missed=1, pen_saved=0),
        _pen_player(3, "GK_B", "Goalkeeper", 900, pen_saved=0, pen_scored=0, pen_missed=0),
        _pen_player(4, "FWD_B", "Attacker", 900, pen_scored=1, pen_missed=4, pen_saved=0),
    ]


def _three_team_exp_minutes() -> pd.DataFrame:
    """Teams A and B have matched players; Team C has no penalty data."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "team": ["A", "A", "B", "B", "C"],
            "exp_minutes": [80.0, 60.0, 80.0, 60.0, 70.0],
        }
    )


# ---------------------------------------------------------------------------
# parse_penalty_stats
# ---------------------------------------------------------------------------


class TestParsePenaltyStats:
    def test_output_columns(self):
        response = [_pen_player(1, "P", "Attacker", 900, pen_scored=3, pen_missed=1, pen_saved=0)]
        df = parse_penalty_stats(response, league_id=39)
        assert set(df.columns) == {
            "player_id", "name", "position", "minutes",
            "pen_scored", "pen_missed", "pen_saved",
        }

    def test_extracts_pen_scored(self):
        response = [_pen_player(1, "P", "Attacker", 900, pen_scored=3, pen_missed=1, pen_saved=0)]
        df = parse_penalty_stats(response, league_id=39)
        assert df.iloc[0]["pen_scored"] == 3

    def test_extracts_pen_missed(self):
        response = [_pen_player(1, "P", "Attacker", 900, pen_scored=3, pen_missed=1, pen_saved=0)]
        df = parse_penalty_stats(response, league_id=39)
        assert df.iloc[0]["pen_missed"] == 1

    def test_extracts_pen_saved(self):
        response = [
            _pen_player(10, "GK", "Goalkeeper", 900, pen_saved=4, pen_scored=0, pen_missed=0)
        ]
        df = parse_penalty_stats(response, league_id=39)
        assert df.iloc[0]["pen_saved"] == 4

    def test_null_pen_scored_treated_as_zero(self):
        response = [
            _pen_player(1, "P", "Attacker", 900, pen_scored=None, pen_missed=None, pen_saved=None)
        ]
        df = parse_penalty_stats(response, league_id=39)
        assert df.iloc[0]["pen_scored"] == 0
        assert df.iloc[0]["pen_missed"] == 0
        assert df.iloc[0]["pen_saved"] == 0

    def test_drops_players_below_min_minutes(self):
        response = [
            _pen_player(1, "P", "Attacker", 900, pen_scored=2),
            _pen_player(2, "Q", "Attacker", 400, pen_scored=5),
        ]
        df = parse_penalty_stats(response, league_id=39, min_minutes=450)
        assert list(df["player_id"]) == [1]

    def test_respects_custom_min_minutes(self):
        response = [_pen_player(1, "P", "Attacker", 300, pen_scored=2)]
        df = parse_penalty_stats(response, league_id=39, min_minutes=200)
        assert len(df) == 1

    def test_forward_position_canonicalized_to_attacker(self):
        response = [_pen_player(1, "P", "Forward", 900, pen_scored=2)]
        df = parse_penalty_stats(response, league_id=39)
        assert df.iloc[0]["position"] == "Attacker"

    def test_empty_response_returns_empty_dataframe(self):
        df = parse_penalty_stats([], league_id=39)
        assert df.empty
        assert "pen_saved" in df.columns

    def test_all_below_min_minutes_returns_empty(self):
        response = [_pen_player(1, "P", "Attacker", 100, pen_scored=2)]
        df = parse_penalty_stats(response, league_id=39, min_minutes=450)
        assert df.empty


# ---------------------------------------------------------------------------
# team_shootout_z
# ---------------------------------------------------------------------------


class TestTeamShootoutZ:
    @pytest.fixture()
    def result(self) -> pd.DataFrame:
        return team_shootout_z(
            {39: _two_team_responses()},
            _three_team_exp_minutes(),
        )

    def test_output_columns(self, result):
        assert set(result.columns) == {"team", "shootout_z"}

    def test_three_rows_for_three_teams(self, result):
        assert set(result["team"]) == {"A", "B", "C"}

    def test_strong_team_higher_than_weak(self, result):
        z_a = float(result.loc[result["team"] == "A", "shootout_z"].iloc[0])
        z_b = float(result.loc[result["team"] == "B", "shootout_z"].iloc[0])
        assert z_a > z_b

    def test_no_data_team_is_zero(self, result):
        z_c = float(result.loc[result["team"] == "C", "shootout_z"].iloc[0])
        assert z_c == pytest.approx(0.0, abs=1e-10)

    def test_z_score_mean_zero_across_teams_with_data(self, result):
        with_data = result[result["team"].isin(["A", "B"])]
        assert float(with_data["shootout_z"].mean()) == pytest.approx(0.0, abs=1e-10)

    def test_strong_team_positive_z(self, result):
        z_a = float(result.loc[result["team"] == "A", "shootout_z"].iloc[0])
        assert z_a > 0.0

    def test_weak_team_negative_z(self, result):
        z_b = float(result.loc[result["team"] == "B", "shootout_z"].iloc[0])
        assert z_b < 0.0

    def test_empty_penalty_by_league_all_zero(self):
        exp = pd.DataFrame(
            {
                "player_id": [1, 2],
                "team": ["X", "Y"],
                "exp_minutes": [80.0, 80.0],
            }
        )
        result = team_shootout_z({}, exp)
        assert set(result["team"]) == {"X", "Y"}
        assert (result["shootout_z"] == 0.0).all()

    def test_dedup_keeps_max_minutes_player(self):
        """Player in two leagues: only the higher-minutes row is used."""
        p_low = _pen_player(1, "P", "Goalkeeper", 600, pen_saved=0)
        p_high = _pen_player(1, "P", "Goalkeeper", 900, pen_saved=10)
        exp = pd.DataFrame({"player_id": [1], "team": ["X"], "exp_minutes": [80.0]})
        result = team_shootout_z({1: [p_high], 2: [p_low]}, exp)
        # Should use pen_saved=10 (from 900-minute row); result is a single team so z=0.
        assert len(result) == 1

    def test_taker_with_no_attempts_is_neutral(self):
        """A team whose only outfield player has no penalty attempts → taker_signal = 0."""
        response = [
            _pen_player(1, "GK", "Goalkeeper", 900, pen_saved=3),
            _pen_player(2, "FWD", "Attacker", 900, pen_scored=0, pen_missed=0),
        ]
        exp = pd.DataFrame({"player_id": [1, 2], "team": ["X", "X"], "exp_minutes": [80.0, 60.0]})
        result = team_shootout_z({39: response}, exp)
        assert len(result) == 1

    def test_single_team_with_data_z_is_zero(self):
        """Single team in exp_minutes with data: z-score of one value → 0."""
        response = [_pen_player(1, "GK", "Goalkeeper", 900, pen_saved=3)]
        exp = pd.DataFrame({"player_id": [1], "team": ["Solo"], "exp_minutes": [80.0]})
        result = team_shootout_z({39: response}, exp)
        solo_z = float(result.loc[result["team"] == "Solo", "shootout_z"].iloc[0])
        assert solo_z == pytest.approx(0.0, abs=1e-10)

    def test_empty_exp_minutes_returns_empty(self):
        exp = pd.DataFrame(columns=["player_id", "team", "exp_minutes"])
        result = team_shootout_z({39: _two_team_responses()}, exp)
        assert result.empty or len(result) == 0
