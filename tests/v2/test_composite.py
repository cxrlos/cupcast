"""Tests for cupcast.v2.clubform.composite — in-memory fixtures, no network."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cupcast.v2.clubform.composite import (
    assemble_player_quality,
    squad_profiles,
    style_profile,
)

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _player(
    player_id: int,
    name: str,
    position: str,
    minutes: int,
    *,
    goals_total: int = 0,
    goals_assists: int = 0,
    goals_saves: int = 0,
    goals_conceded: int = 0,
    shots_on: int = 0,
    passes_key: int = 0,
    tackles_total: int = 0,
    tackles_blocks: int = 0,
    tackles_interceptions: int = 0,
    duels_won: int = 0,
) -> dict:
    return {
        "player": {"id": player_id, "name": name},
        "statistics": [
            {
                "games": {
                    "minutes": minutes,
                    "position": position,
                    "rating": "7.00",
                    "appearences": 10,
                    "lineups": 9,
                },
                "goals": {
                    "total": goals_total,
                    "assists": goals_assists,
                    "saves": goals_saves,
                    "conceded": goals_conceded,
                },
                "shots": {"total": shots_on, "on": shots_on},
                "passes": {"total": 200, "key": passes_key},
                "tackles": {
                    "total": tackles_total,
                    "blocks": tackles_blocks,
                    "interceptions": tackles_interceptions,
                },
                "duels": {"total": duels_won * 2, "won": duels_won},
            }
        ],
    }


def _quality_df() -> pd.DataFrame:
    """Six fully-specified quality rows for two teams.

    Brazil: p1 (Att), p2 (Mid), p3 (Def), p4 (GK).
    Germany: p5 (Att), p6 (Mid).
    Player 99 intentionally absent — used as an "unmatched" player in exp_minutes.
    """
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 6],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
            "position": [
                "Attacker",
                "Midfielder",
                "Defender",
                "Goalkeeper",
                "Attacker",
                "Midfielder",
            ],
            "minutes": [900, 810, 720, 900, 800, 700],
            "att_z": [1.0, 0.5, -0.5, -0.8, 0.8, 0.2],
            "def_z": [-0.2, 0.8, 1.0, 0.1, -0.1, 0.7],
            "gk_z": [
                float("nan"),
                float("nan"),
                float("nan"),
                0.6,
                float("nan"),
                float("nan"),
            ],
            "save_rate": [
                float("nan"),
                float("nan"),
                float("nan"),
                0.75,
                float("nan"),
                float("nan"),
            ],
        }
    )


def _exp_minutes_df() -> pd.DataFrame:
    """Brazil: p1+p2+p3+p4 (matched) + p99 (unmatched). Germany: p5+p6 (all matched)."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 99, 5, 6],
            "team": [
                "Brazil",
                "Brazil",
                "Brazil",
                "Brazil",
                "Brazil",
                "Germany",
                "Germany",
            ],
            "exp_minutes": [80.0, 60.0, 70.0, 65.0, 50.0, 75.0, 55.0],
        }
    )


def _p20(values: list[float]) -> float:
    return float(np.percentile(values, 20))


# 20th-percentile floor for the quality df above (used in several assertions).
_P20_ATT = _p20([1.0, 0.5, -0.5, -0.8, 0.8, 0.2])  # ≈ -0.5
_P20_DEF = _p20([-0.2, 0.8, 1.0, 0.1, -0.1, 0.7])  # ≈ -0.1


# ---------------------------------------------------------------------------
# assemble_player_quality
# ---------------------------------------------------------------------------


class TestAssemblePlayerQuality:
    def test_returns_expected_columns(self):
        resp = [_player(1, "Alice", "Attacker", 900, goals_total=10)]
        pq = assemble_player_quality({39: resp}, {39: 1.0})
        required = {
            "player_id", "name", "position", "minutes",
            "att_z", "def_z", "gk_z", "save_rate",
        }
        assert required.issubset(set(pq.columns))

    def test_drops_players_below_min_minutes(self):
        resp = [
            _player(1, "Alice", "Attacker", 900),
            _player(2, "Bob", "Midfielder", 400),
        ]
        pq = assemble_player_quality({39: resp}, {39: 1.0})
        assert set(pq["player_id"]) == {1}

    def test_deduplicates_keeps_higher_minutes_row(self):
        # Player 1 in two leagues; keep the row from league 140 (more minutes).
        resp_a = [_player(1, "Alice", "Attacker", 900)]
        resp_b = [_player(1, "Alice", "Attacker", 1200)]
        pq = assemble_player_quality({39: resp_a, 140: resp_b}, {39: 1.0, 140: 1.0})
        rows = pq[pq["player_id"] == 1]
        assert len(rows) == 1
        assert float(rows["minutes"].iloc[0]) == 1200.0

    def test_gk_att_z_scored_within_gk_group_only(self):
        """GKs score 0 goals; within the GK group they have zero variance → att_z = 0.

        If pooled with outfield players the GKs would receive att_z << -1.
        """
        resp = [
            _player(1, "GK1", "Goalkeeper", 900, goals_saves=50, goals_conceded=20),
            _player(2, "GK2", "Goalkeeper", 900, goals_saves=30, goals_conceded=30),
            _player(3, "ATT1", "Attacker", 900, goals_total=15),
            _player(4, "ATT2", "Attacker", 900, goals_total=5),
        ]
        pq = assemble_player_quality({39: resp}, {39: 1.0})
        gk_att_z = pq.loc[pq["position"] == "Goalkeeper", "att_z"]
        # Both GKs have att_raw ≈ 0; within-GK z-score = 0.
        assert gk_att_z.abs().max() == pytest.approx(0.0, abs=1e-10)

    def test_gk_z_nan_for_non_gk(self):
        resp = [
            _player(1, "Alice", "Attacker", 900, goals_total=10),
            _player(2, "GK", "Goalkeeper", 900, goals_saves=40, goals_conceded=15),
        ]
        pq = assemble_player_quality({39: resp}, {39: 1.0})
        att_row = pq[pq["player_id"] == 1].iloc[0]
        assert math.isnan(att_row["gk_z"])

    def test_save_rate_set_for_gk_only(self):
        resp = [
            _player(1, "GK", "Goalkeeper", 900, goals_saves=50, goals_conceded=20),
            _player(2, "Alice", "Attacker", 900, goals_total=10),
        ]
        pq = assemble_player_quality({39: resp}, {39: 1.0})
        assert not math.isnan(pq.loc[pq["player_id"] == 1, "save_rate"].iloc[0])
        assert math.isnan(pq.loc[pq["player_id"] == 2, "save_rate"].iloc[0])

    def test_higher_league_strength_shifts_z_scores(self):
        """With two players in the same position, the stronger scorer gets att_z > 0."""
        resp = [
            _player(1, "P", "Attacker", 900, goals_total=10),
            _player(2, "Q", "Attacker", 900, goals_total=0),
        ]
        pq = assemble_player_quality({39: resp}, {39: 1.5})
        assert pq.loc[pq["player_id"] == 1, "att_z"].iloc[0] > 0

    def test_empty_response_returns_empty_dataframe(self):
        pq = assemble_player_quality({}, {})
        assert pq.empty
        assert "att_z" in pq.columns


# ---------------------------------------------------------------------------
# squad_profiles
# ---------------------------------------------------------------------------


class TestSquadProfiles:
    @pytest.fixture()
    def profiles(self) -> pd.DataFrame:
        return squad_profiles(_quality_df(), _exp_minutes_df())

    def test_one_row_per_team(self, profiles):
        assert set(profiles["team"]) == {"Brazil", "Germany"}

    def test_required_columns(self, profiles):
        assert {"team", "attack", "defense", "gk", "coverage", "n_players"}.issubset(
            set(profiles.columns)
        )

    # --- position-aware aggregation ---

    def test_gk_att_z_excluded_from_team_attack(self, profiles):
        """Brazil GK (player 4, att_z=-0.8) must NOT enter the attack composite.

        Attack = weighted mean over Attacker + Midfielder + unmatched (floor-imputed).
        """
        brazil = profiles[profiles["team"] == "Brazil"].iloc[0]
        # p1 (Att, em=80), p2 (Mid, em=60), p99 (unmatched, em=50, att_z=p20)
        expected = (1.0 * 80 + 0.5 * 60 + _P20_ATT * 50) / (80 + 60 + 50)
        assert brazil["attack"] == pytest.approx(expected, rel=1e-6)

    def test_attacker_excluded_from_defense(self, profiles):
        """Brazil Attacker (player 1) must NOT enter the defense composite.

        Defense = weighted mean over Defender + Midfielder + unmatched.
        """
        brazil = profiles[profiles["team"] == "Brazil"].iloc[0]
        # p2 (Mid, def_z=0.8, em=60), p3 (Def, def_z=1.0, em=70), p99 (unmatched, em=50)
        expected = (0.8 * 60 + 1.0 * 70 + _P20_DEF * 50) / (60 + 70 + 50)
        assert brazil["defense"] == pytest.approx(expected, rel=1e-6)

    def test_gk_composite_from_goalkeeper_only(self, profiles):
        """GK composite = weighted mean of gk_z over Goalkeepers only."""
        brazil = profiles[profiles["team"] == "Brazil"].iloc[0]
        # p4 (GK, gk_z=0.6, em=65)
        assert brazil["gk"] == pytest.approx(0.6, abs=1e-10)

    def test_gk_nan_when_no_matched_goalkeeper(self, profiles):
        germany = profiles[profiles["team"] == "Germany"].iloc[0]
        # Germany has no GK in exp_minutes.
        assert math.isnan(germany["gk"])

    # --- minutes-weighting ---

    def test_higher_exp_minutes_more_influence(self, profiles):
        """Germany: p5 (Att, att_z=0.8, em=75) and p6 (Mid, att_z=0.2, em=55)."""
        germany = profiles[profiles["team"] == "Germany"].iloc[0]
        expected = (0.8 * 75 + 0.2 * 55) / (75 + 55)
        assert germany["attack"] == pytest.approx(expected, rel=1e-6)

    # --- coverage ---

    def test_coverage_below_one_when_unmatched_player_present(self, profiles):
        brazil = profiles[profiles["team"] == "Brazil"].iloc[0]
        # matched em = 80+60+70+65 = 275; total em = 275+50 = 325
        assert brazil["coverage"] == pytest.approx(275 / 325, rel=1e-6)

    def test_coverage_full_when_all_players_matched(self, profiles):
        germany = profiles[profiles["team"] == "Germany"].iloc[0]
        assert germany["coverage"] == pytest.approx(1.0, abs=1e-10)

    def test_n_players_includes_unmatched(self, profiles):
        brazil = profiles[profiles["team"] == "Brazil"].iloc[0]
        germany = profiles[profiles["team"] == "Germany"].iloc[0]
        assert int(brazil["n_players"]) == 5
        assert int(germany["n_players"]) == 2

    # --- floor imputation drags composite down ---

    def test_unmatched_player_drags_attack_down(self):
        """Adding an unmatched player (floor-imputed) reduces the team attack composite."""
        quality = _quality_df()
        em_clean = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "team": ["Brazil"] * 4,
                "exp_minutes": [80.0, 60.0, 70.0, 65.0],
            }
        )
        em_with_gap = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 99],
                "team": ["Brazil"] * 5,
                "exp_minutes": [80.0, 60.0, 70.0, 65.0, 50.0],
            }
        )
        attack_clean = squad_profiles(quality, em_clean).loc[0, "attack"]
        attack_gapped = squad_profiles(quality, em_with_gap).loc[0, "attack"]
        assert attack_gapped < attack_clean


# ---------------------------------------------------------------------------
# style_profile
# ---------------------------------------------------------------------------


def _lineup_entry(player_id: int, pos: str) -> dict:
    return {
        "player": {
            "id": player_id,
            "name": f"P{player_id}",
            "pos": pos,
            "grid": "1:1",
        }
    }


def _block(team: dict, formation: str, g: int, d: list, m: list, f: list) -> dict:
    start_xi = (
        [_lineup_entry(g, "G")]
        + [_lineup_entry(pid, "D") for pid in d]
        + [_lineup_entry(pid, "M") for pid in m]
        + [_lineup_entry(pid, "F") for pid in f]
    )
    return {
        "team": team,
        "formation": formation,
        "startXI": start_xi,
        "substitutes": [],
    }


def _fixtures_433_433_442() -> dict[int, list[dict]]:
    """Three fixtures for Brazil: two 4-3-3 and one 4-4-2. Argentina is a filler opponent."""
    brazil = {"id": 10, "name": "Brazil"}
    argentina = {"id": 20, "name": "Argentina"}
    brazil_433 = _block(brazil, "4-3-3", 1, [2, 3, 4, 5], [6, 7, 8], [9, 10, 11])
    brazil_442 = _block(brazil, "4-4-2", 1, [2, 3, 4, 5], [6, 7, 8, 9], [10, 11])
    argentina_433 = _block(argentina, "4-3-3", 20, [21, 22, 23, 24], [25, 26, 27], [28, 29, 30])
    return {
        1: [brazil_433, argentina_433],
        2: [brazil_433, argentina_433],
        3: [brazil_442, argentina_433],
    }


class TestStyleProfile:
    @pytest.fixture()
    def profile(self) -> dict:
        return style_profile(_fixtures_433_433_442(), "Brazil")

    def test_has_required_top_level_keys(self, profile):
        assert "formations" in profile
        assert "lines" in profile

    def test_formations_sum_to_one(self, profile):
        assert sum(profile["formations"].values()) == pytest.approx(1.0, abs=1e-10)

    def test_formation_frequencies_correct(self, profile):
        assert profile["formations"]["4-3-3"] == pytest.approx(2 / 3, rel=1e-6)
        assert profile["formations"]["4-4-2"] == pytest.approx(1 / 3, rel=1e-6)

    def test_lines_has_all_keys(self, profile):
        assert {"G", "D", "M", "F"}.issubset(set(profile["lines"]))

    def test_lines_g_mean_is_one(self, profile):
        # Every fixture has exactly 1 GK.
        assert profile["lines"]["G"] == pytest.approx(1.0)

    def test_lines_d_mean_is_four(self, profile):
        # Every fixture has 4 defenders.
        assert profile["lines"]["D"] == pytest.approx(4.0)

    def test_lines_m_mean_correct(self, profile):
        # 4-3-3: 3 mid, 4-3-3: 3 mid, 4-4-2: 4 mid → mean = 10/3
        assert profile["lines"]["M"] == pytest.approx(10 / 3, rel=1e-6)

    def test_lines_f_mean_correct(self, profile):
        # 4-3-3: 3, 4-3-3: 3, 4-4-2: 2 → mean = 8/3
        assert profile["lines"]["F"] == pytest.approx(8 / 3, rel=1e-6)

    def test_unknown_team_returns_empty(self):
        result = style_profile(_fixtures_433_433_442(), "UnknownFC")
        assert result["formations"] == {}
        assert result["lines"] == {"D": 0.0, "M": 0.0, "F": 0.0, "G": 0.0}

    def test_opponent_lineups_not_mixed_into_team_profile(self):
        """Argentina appears in every fixture but must not affect Brazil's lines."""
        profile = style_profile(_fixtures_433_433_442(), "Brazil")
        # In our fixture, Argentina plays 4-3-3 in all 3 matches.
        # Brazil's profile should show both 4-3-3 and 4-4-2, not only 4-3-3.
        assert "4-4-2" in profile["formations"]

    def test_empty_fixture_dict_returns_empty(self):
        result = style_profile({}, "Brazil")
        assert result["formations"] == {}
