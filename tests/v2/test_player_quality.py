"""Tests for cupcast.v2.clubform.player_quality — in-memory fixtures only."""

import math

import pandas as pd
import pytest

from cupcast.v2.clubform.player_quality import (
    apply_league_strength,
    apply_understat_xg,
    parse_player_stats,
    zscore_within_position,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _player(
    player_id: int,
    name: str,
    position: str,
    minutes: int,
    *,
    rating: str = "7.00",
    goals_total=0,
    goals_assists=0,
    goals_saves=None,
    goals_conceded=None,
    shots_on=0,
    passes_key=0,
    tackles_total=0,
    tackles_blocks=0,
    tackles_interceptions=0,
    duels_won=0,
) -> dict:
    return {
        "player": {"id": player_id, "name": name},
        "statistics": [
            {
                "games": {
                    "minutes": minutes,
                    "position": position,
                    "rating": rating,
                    "appearences": 10,
                    "lineups": 9,
                },
                "goals": {
                    "total": goals_total,
                    "assists": goals_assists,
                    "saves": goals_saves,
                    "conceded": goals_conceded,
                },
                "shots": {
                    "total": None if shots_on is None else shots_on + 2,
                    "on": shots_on,
                },
                "passes": {
                    "total": None if passes_key is None else passes_key * 5,
                    "key": passes_key,
                },
                "tackles": {
                    "total": tackles_total,
                    "blocks": tackles_blocks,
                    "interceptions": tackles_interceptions,
                },
                "duels": {
                    "total": None if duels_won is None else duels_won * 2,
                    "won": duels_won,
                },
            }
        ],
    }


def _base_response() -> list[dict]:
    """
    Two Attackers, two Goalkeepers, one below-threshold Attacker, one null-stats
    Midfielder.  Only the first four (plus the Midfielder) pass min_minutes=450.

    Attacker 1 (id=1, 900 min):
        att_raw = (5+3)/10 + 0.5*8/10 + 0.3*20/10 = 0.8 + 0.4 + 0.6 = 1.8
        def_raw = (10+3+2+15)/10 = 3.0

    Attacker 2 (id=2, 900 min):
        att_raw = (2+1)/10 + 0.5*4/10 + 0.3*10/10 = 0.3 + 0.2 + 0.3 = 0.8
        def_raw = (5+1+1+8)/10 = 1.5

    Goalkeeper 1 (id=3, 900 min): gk_raw = 35/10 = 3.5, save_rate = 35/50 = 0.70
    Goalkeeper 2 (id=4, 900 min): gk_raw = 20/10 = 2.0, save_rate = 20/40 = 0.50

    Low-minutes Attacker (id=5, 400 min): dropped by parse_player_stats.
    Null-stats Midfielder (id=6, 500 min): all stats coerce to 0; gk_raw = NaN.
    """
    return [
        _player(
            1, "Striker One", "Attacker", 900,
            goals_total=5, goals_assists=3, shots_on=8, passes_key=20,
            tackles_total=10, tackles_blocks=2, tackles_interceptions=3, duels_won=15,
        ),
        _player(
            2, "Striker Two", "Attacker", 900,
            goals_total=2, goals_assists=1, shots_on=4, passes_key=10,
            tackles_total=5, tackles_blocks=1, tackles_interceptions=1, duels_won=8,
        ),
        _player(
            3, "Keeper One", "Goalkeeper", 900,
            goals_saves=35, goals_conceded=15,
        ),
        _player(
            4, "Keeper Two", "Goalkeeper", 900,
            goals_saves=20, goals_conceded=20,
        ),
        _player(5, "Bench Striker", "Attacker", 400),  # below min_minutes
        _player(
            6, "Mid Null", "Midfielder", 500,
            goals_total=None, goals_assists=None, goals_saves=None,
            goals_conceded=None, shots_on=None, passes_key=None,
            tackles_total=None, tackles_blocks=None, tackles_interceptions=None,
            duels_won=None,
        ),
    ]


def _parsed(min_minutes: int = 450) -> pd.DataFrame:
    return parse_player_stats(_base_response(), league_id=39, min_minutes=min_minutes)


# ---------------------------------------------------------------------------
# parse_player_stats
# ---------------------------------------------------------------------------


class TestParsePlayerStats:
    def test_output_columns(self):
        df = _parsed()
        required = {"player_id", "name", "position", "minutes", "rating",
                    "att_raw", "def_raw", "gk_raw", "save_rate"}
        assert required.issubset(set(df.columns))

    def test_row_count_respects_min_minutes(self):
        # 5 players pass 450; player id=5 (400 min) is dropped.
        assert len(_parsed()) == 5

    def test_min_minutes_drop(self):
        df = _parsed()
        assert 5 not in df["player_id"].values

    def test_attacker_att_raw(self):
        df = _parsed()
        row = df[df["player_id"] == 1].iloc[0]
        assert row["att_raw"] == pytest.approx(1.8)

    def test_attacker_def_raw(self):
        df = _parsed()
        row = df[df["player_id"] == 1].iloc[0]
        assert row["def_raw"] == pytest.approx(3.0)

    def test_attacker_gk_raw_is_nan(self):
        df = _parsed()
        row = df[df["player_id"] == 1].iloc[0]
        assert math.isnan(row["gk_raw"])

    def test_attacker_save_rate_is_nan(self):
        df = _parsed()
        row = df[df["player_id"] == 1].iloc[0]
        assert math.isnan(row["save_rate"])

    def test_goalkeeper_gk_raw(self):
        df = _parsed()
        row = df[df["player_id"] == 3].iloc[0]
        assert row["gk_raw"] == pytest.approx(3.5)

    def test_goalkeeper_save_rate(self):
        df = _parsed()
        row = df[df["player_id"] == 3].iloc[0]
        assert row["save_rate"] == pytest.approx(0.7)

    def test_goalkeeper_save_rate_second(self):
        df = _parsed()
        row = df[df["player_id"] == 4].iloc[0]
        assert row["save_rate"] == pytest.approx(0.5)

    def test_rating_parsed_as_float(self):
        df = _parsed()
        assert df["rating"].dtype.kind == "f"

    def test_null_stats_no_crash(self):
        df = _parsed()
        mid = df[df["player_id"] == 6].iloc[0]
        assert mid["att_raw"] == pytest.approx(0.0)
        assert mid["def_raw"] == pytest.approx(0.0)

    def test_null_stats_gk_raw_is_nan(self):
        df = _parsed()
        mid = df[df["player_id"] == 6].iloc[0]
        assert math.isnan(mid["gk_raw"])

    def test_gk_zero_denominator_save_rate_nan(self):
        response = [_player(99, "Ghost GK", "Goalkeeper", 900, goals_saves=0, goals_conceded=0)]
        df = parse_player_stats(response, league_id=1)
        row = df[df["player_id"] == 99].iloc[0]
        assert math.isnan(row["save_rate"])

    def test_second_attacker_att_raw(self):
        df = _parsed()
        row = df[df["player_id"] == 2].iloc[0]
        assert row["att_raw"] == pytest.approx(0.8)

    def test_forward_position_canonicalized_to_attacker(self):
        response = [
            _player(
                1, "F", "Forward", 900,
                goals_total=5, goals_assists=2, shots_on=6, passes_key=8,
                tackles_total=5, tackles_blocks=0, tackles_interceptions=2, duels_won=20,
            ),
        ]
        df = parse_player_stats(response, league_id=39, min_minutes=450)
        assert df.iloc[0]["position"] == "Attacker"


# ---------------------------------------------------------------------------
# apply_league_strength
# ---------------------------------------------------------------------------


class TestApplyLeagueStrength:
    def test_att_raw_scaled(self):
        df = _parsed()
        out = apply_league_strength(df, league_id=39, strength_index={39: 1.2})
        row = out[out["player_id"] == 1].iloc[0]
        assert row["att_raw"] == pytest.approx(1.8 * 1.2)

    def test_def_raw_scaled(self):
        df = _parsed()
        out = apply_league_strength(df, league_id=39, strength_index={39: 1.2})
        row = out[out["player_id"] == 1].iloc[0]
        assert row["def_raw"] == pytest.approx(3.0 * 1.2)

    def test_gk_raw_not_scaled(self):
        df = _parsed()
        out = apply_league_strength(df, league_id=39, strength_index={39: 1.2})
        row = out[out["player_id"] == 3].iloc[0]
        assert row["gk_raw"] == pytest.approx(3.5)

    def test_save_rate_not_scaled(self):
        df = _parsed()
        out = apply_league_strength(df, league_id=39, strength_index={39: 1.2})
        row = out[out["player_id"] == 3].iloc[0]
        assert row["save_rate"] == pytest.approx(0.7)

    def test_missing_league_id_falls_back_to_one(self):
        df = _parsed()
        original_att = df[df["player_id"] == 1].iloc[0]["att_raw"]
        out = apply_league_strength(df, league_id=39, strength_index={99: 1.5})
        assert out[out["player_id"] == 1].iloc[0]["att_raw"] == pytest.approx(original_att)

    def test_returns_copy(self):
        df = _parsed()
        out = apply_league_strength(df, league_id=39, strength_index={39: 1.2})
        assert out is not df


# ---------------------------------------------------------------------------
# zscore_within_position
# ---------------------------------------------------------------------------


class TestZscoreWithinPosition:
    def test_output_columns_added(self):
        df = zscore_within_position(_parsed())
        assert {"att_z", "def_z", "gk_z"}.issubset(set(df.columns))

    def test_attacker_att_z_positive_for_stronger_player(self):
        # att_raw = [1.8, 0.8]; mean=1.3, std=0.5 (ddof=0) → z1=1.0
        df = zscore_within_position(_parsed())
        z = df[df["player_id"] == 1].iloc[0]["att_z"]
        assert z == pytest.approx(1.0)

    def test_attacker_att_z_negative_for_weaker_player(self):
        df = zscore_within_position(_parsed())
        z = df[df["player_id"] == 2].iloc[0]["att_z"]
        assert z == pytest.approx(-1.0)

    def test_att_z_sum_zero_within_group(self):
        df = zscore_within_position(_parsed())
        att_z = df[df["position"] == "Attacker"]["att_z"]
        assert att_z.sum() == pytest.approx(0.0, abs=1e-10)

    def test_gk_gk_z_positive_for_more_saves(self):
        # gk_raw = [3.5, 2.0]; mean=2.75, std=0.75 (ddof=0) → z_gk1=1.0
        df = zscore_within_position(_parsed())
        z = df[df["player_id"] == 3].iloc[0]["gk_z"]
        assert z == pytest.approx(1.0)

    def test_gk_gk_z_negative_for_fewer_saves(self):
        df = zscore_within_position(_parsed())
        z = df[df["player_id"] == 4].iloc[0]["gk_z"]
        assert z == pytest.approx(-1.0)

    def test_non_gk_gk_z_is_nan(self):
        df = zscore_within_position(_parsed())
        for pid in (1, 2, 6):
            val = df[df["player_id"] == pid].iloc[0]["gk_z"]
            assert math.isnan(val), f"player {pid}: expected NaN gk_z, got {val}"

    def test_single_position_group_z_zero(self):
        # Midfielder (id=6) is the only player in its group → z = 0
        df = zscore_within_position(_parsed())
        row = df[df["player_id"] == 6].iloc[0]
        assert row["att_z"] == pytest.approx(0.0)
        assert row["def_z"] == pytest.approx(0.0)

    def test_returns_copy(self):
        df = _parsed()
        out = zscore_within_position(df)
        assert out is not df


# ---------------------------------------------------------------------------
# apply_understat_xg
# ---------------------------------------------------------------------------


class TestApplyUnderstatXg:
    def _understat(self) -> pd.DataFrame:
        return pd.DataFrame({"player": ["Striker One"], "quality_per90": [0.95]})

    def test_replaces_att_raw_for_matched_player(self):
        df = _parsed()
        out = apply_understat_xg(df, self._understat())
        row = out[out["player_id"] == 1].iloc[0]
        assert row["att_raw"] == pytest.approx(0.95)

    def test_unmatched_player_unchanged(self):
        df = _parsed()
        original_att = df[df["player_id"] == 2].iloc[0]["att_raw"]
        out = apply_understat_xg(df, self._understat())
        assert out[out["player_id"] == 2].iloc[0]["att_raw"] == pytest.approx(original_att)

    def test_noop_on_none(self):
        df = _parsed()
        original = df["att_raw"].copy()
        out = apply_understat_xg(df, None)
        pd.testing.assert_series_equal(out["att_raw"], original)

    def test_noop_on_empty(self):
        df = _parsed()
        original = df["att_raw"].copy()
        out = apply_understat_xg(df, pd.DataFrame())
        pd.testing.assert_series_equal(out["att_raw"], original)

    def test_accent_normalized_match(self):
        understat = pd.DataFrame({"player": ["Ángel Torres"], "quality_per90": [0.60]})
        response = [
            _player(10, "Ángel Torres", "Attacker", 900, goals_total=3),
            _player(11, "Other Player", "Attacker", 900, goals_total=1),
        ]
        df = parse_player_stats(response, league_id=1)
        out = apply_understat_xg(df, understat)
        assert out[out["player_id"] == 10].iloc[0]["att_raw"] == pytest.approx(0.60)
        # Other player must be unchanged.
        assert out[out["player_id"] == 11].iloc[0]["att_raw"] != pytest.approx(0.60)
