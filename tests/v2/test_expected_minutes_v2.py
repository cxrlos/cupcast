"""Tests for cupcast.v2.clubform.expected_minutes — in-memory fixtures only."""

import pandas as pd
import pytest

from cupcast.v2.clubform.expected_minutes import (
    expected_minutes,
    national_team_appearances,
)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

_BRAZIL = {"id": 10, "name": "Brazil"}
_ARGENTINA = {"id": 20, "name": "Argentina"}


def _entry(player_id: int) -> dict:
    return {"player": {"id": player_id, "name": f"Player {player_id}", "pos": "M", "grid": "4:1"}}


def _block(team: dict, start_ids: list[int], sub_ids: list[int]) -> dict:
    return {
        "team": team,
        "formation": "4-3-3",
        "startXI": [_entry(pid) for pid in start_ids],
        "substitutes": [_entry(pid) for pid in sub_ids],
        "coach": {"id": 999, "name": "Coach"},
    }


# 3 fixtures spanning > 1 year — fixture 1 is oldest, fixture 3 is newest.
#
# Player roles (Brazil):
#   1 = STARTER  — startXI in all 3 fixtures
#   2 = SUB      — substitutes in fixtures 2 and 3 only
#   3 = ONE_OFF  — startXI in fixture 1 only (oldest, low decay weight)
#   4 = RECENT   — startXI in fixture 3 only (newest, weight = 1.0)
#  99 = Argentina filler
_FIXTURE_DATES = {
    1: "2024-01-01",
    2: "2024-06-01",
    3: "2025-06-01",
}
_LINEUPS = {
    1: [
        _block(_BRAZIL, start_ids=[1, 3], sub_ids=[]),
        _block(_ARGENTINA, start_ids=[99], sub_ids=[]),
    ],
    2: [
        _block(_BRAZIL, start_ids=[1], sub_ids=[2]),
        _block(_ARGENTINA, start_ids=[99], sub_ids=[]),
    ],
    3: [
        _block(_BRAZIL, start_ids=[1, 4], sub_ids=[2]),
        _block(_ARGENTINA, start_ids=[99], sub_ids=[]),
    ],
}
_AS_OF = "2025-06-01"


@pytest.fixture()
def apps() -> pd.DataFrame:
    return national_team_appearances(_LINEUPS, _FIXTURE_DATES, as_of=_AS_OF)


@pytest.fixture()
def exp(apps: pd.DataFrame) -> pd.DataFrame:
    return expected_minutes(apps)


def _row(df: pd.DataFrame, player_id: int) -> pd.Series:
    return df[df["player_id"] == player_id].iloc[0]


# ---------------------------------------------------------------------------
# national_team_appearances — columns and types
# ---------------------------------------------------------------------------


class TestNationalTeamAppearancesColumns:
    def test_required_columns(self, apps):
        required = {"player_id", "team", "w_starts", "w_sub_apps", "w_team_matches"}
        assert required.issubset(set(apps.columns))

    def test_all_brazil_players_present(self, apps):
        assert {1, 2, 3, 4}.issubset(set(apps["player_id"].values))

    def test_team_assigned_to_brazil(self, apps):
        for pid in (1, 2, 3, 4):
            assert _row(apps, pid)["team"] == "Brazil"

    def test_w_team_matches_same_for_all_brazil(self, apps):
        # All Brazil players share the same w_team_matches (3 fixtures of Brazil).
        vals = apps[apps["player_id"].isin([1, 2, 3, 4])]["w_team_matches"]
        assert vals.nunique() == 1


# ---------------------------------------------------------------------------
# Starter vs sub
# ---------------------------------------------------------------------------


class TestStarterVsSub:
    def test_starter_has_higher_w_starts(self, apps):
        assert _row(apps, 1)["w_starts"] > _row(apps, 2)["w_starts"]

    def test_sub_has_zero_w_starts(self, apps):
        assert _row(apps, 2)["w_starts"] == pytest.approx(0.0)

    def test_starter_has_higher_start_prob(self, exp):
        assert _row(exp, 1)["start_prob"] > _row(exp, 2)["start_prob"]

    def test_starter_has_higher_exp_minutes(self, exp):
        assert _row(exp, 1)["exp_minutes"] > _row(exp, 2)["exp_minutes"]

    def test_starter_start_prob_is_one(self, exp):
        # Appeared in startXI every match → start_prob = 1.0
        assert _row(exp, 1)["start_prob"] == pytest.approx(1.0)

    def test_starter_exp_minutes_is_ninety(self, exp):
        assert _row(exp, 1)["exp_minutes"] == pytest.approx(90.0)

    def test_sub_contributes_nonzero_exp_minutes(self, exp):
        # 2 sub appearances → positive exp_minutes via sub_minutes_prior
        assert _row(exp, 2)["exp_minutes"] > 0.0


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------


class TestRecencyWeighting:
    def test_recent_one_off_outweighs_old_one_off(self, apps):
        # Player 4: appeared once in fixture 3 (age = 0, weight = 1.0)
        # Player 3: appeared once in fixture 1 (age > 365 days, weight < 1.0)
        assert _row(apps, 4)["w_starts"] > _row(apps, 3)["w_starts"]

    def test_recent_has_higher_start_prob(self, exp):
        assert _row(exp, 4)["start_prob"] > _row(exp, 3)["start_prob"]

    def test_as_of_none_defaults_to_max_date(self):
        # Omitting as_of must give identical results to passing the known max date.
        without = national_team_appearances(_LINEUPS, _FIXTURE_DATES, as_of=None)
        with_explicit = national_team_appearances(_LINEUPS, _FIXTURE_DATES, as_of=_AS_OF)
        pd.testing.assert_frame_equal(
            without.sort_values("player_id").reset_index(drop=True),
            with_explicit.sort_values("player_id").reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_start_prob_in_unit_interval(self, exp):
        assert (exp["start_prob"] >= 0.0).all()
        assert (exp["start_prob"] <= 1.0).all()

    def test_exp_minutes_nonnegative(self, exp):
        assert (exp["exp_minutes"] >= 0.0).all()

    def test_exp_minutes_at_most_ninety(self, exp):
        assert (exp["exp_minutes"] <= 90.0).all()


# ---------------------------------------------------------------------------
# expected_minutes — column contract
# ---------------------------------------------------------------------------


class TestExpectedMinutesColumns:
    def test_required_columns(self, exp):
        assert {"player_id", "start_prob", "exp_minutes"}.issubset(set(exp.columns))

    def test_no_extra_columns(self, exp):
        assert set(exp.columns) == {"player_id", "start_prob", "exp_minutes"}
