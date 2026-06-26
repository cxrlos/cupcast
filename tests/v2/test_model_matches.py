from __future__ import annotations

import pandas as pd

from cupcast.v2.model.matches import assemble_internationals

_EXPECTED_COLS = [
    "date",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "competition",
    "neutral",
    "host_home",
    "period",
]


def _fx(
    league_id: int, date_str: str, home: str, away: str, hg: int, ag: int, status: str = "FT"
) -> dict:
    return {
        "fixture": {"date": date_str, "status": {"short": status}},
        "league": {"id": league_id, "name": "League", "season": 2022},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "goals": {"home": hg, "away": ag},
    }


class _StubClient:
    def __init__(self, data: dict[tuple[int, int], list[dict]]) -> None:
        self._data = data

    def get_response(self, endpoint: str, params: dict) -> list[dict]:
        league = params.get("league")
        season = params.get("season")
        return self._data.get((league, season), [])


# ---------------------------------------------------------------------------
# Fixtures shared across tests
# ---------------------------------------------------------------------------

_COMP_SEASONS = [(1, 2022), (10, 2023), (37, 2024)]

_DATA: dict[tuple[int, int], list[dict]] = {
    # WC 2022 finals (neutral)
    (1, 2022): [
        _fx(1, "2022-11-20T16:00:00+00:00", "Qatar", "Ecuador", 0, 2),
        _fx(1, "2022-12-18T15:00:00+00:00", "Argentina", "France", 3, 3),
    ],
    # Friendlies 2023 (not neutral) — both teams also appear in an official
    # competition above, so they survive the FIFA-national-team filter.
    (10, 2023): [
        _fx(10, "2023-03-25T20:45:00+00:00", "France", "Brazil", 2, 0),
        # Not finished — should be excluded
        _fx(10, "2023-06-17T17:30:00+00:00", "France", "Greece", 1, 0, status="NS"),
    ],
    # WC Qualifiers 2024 (not neutral) — well after a plausible cutoff
    (37, 2024): [
        _fx(37, "2024-10-15T20:45:00+00:00", "Brazil", "Argentina", 1, 1),
        # Null goals — should be excluded
        {
            "fixture": {"date": "2024-11-19T20:45:00+00:00", "status": {"short": "FT"}},
            "league": {"id": 37, "name": "WC Qual", "season": 2024},
            "teams": {"home": {"name": "Colombia"}, "away": {"name": "Chile"}},
            "goals": {"home": None, "away": None},
        },
    ],
}

_CLIENT = _StubClient(_DATA)


# ---------------------------------------------------------------------------


class TestColumns:
    def test_exact_column_order(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        assert list(df.columns) == _EXPECTED_COLS

    def test_dtypes_goals_are_int(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        assert df["home_goals"].dtype.kind == "i"
        assert df["away_goals"].dtype.kind == "i"


class TestNeutralHost:
    def test_wc_finals_neutral_true(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        wc = df[df["competition"] == 1]
        assert wc["neutral"].all(), "WC-finals rows should be neutral"

    def test_wc_finals_host_home_false(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        wc = df[df["competition"] == 1]
        assert (~wc["host_home"]).all(), "WC-finals rows should not be host_home"

    def test_friendlies_neutral_false(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        fr = df[df["competition"] == 10]
        assert (~fr["neutral"]).all(), "Friendlies rows should not be neutral"

    def test_friendlies_host_home_true(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        fr = df[df["competition"] == 10]
        assert fr["host_home"].all(), "Friendlies rows should be host_home"

    def test_qualifiers_not_neutral(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        q = df[df["competition"] == 37]
        assert (~q["neutral"]).all()
        assert q["host_home"].all()


class TestFiltering:
    def test_non_finished_excluded(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        # "France vs Greece" has status "NS" — must not appear
        assert "Greece" not in df["away"].values

    def test_null_goals_excluded(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        # Colombia vs Chile has null goals
        assert "Colombia" not in df["home"].values

    def test_row_count(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        # 2 (WC) + 1 (Friendly FT) + 1 (Qual with goals) = 4
        assert len(df) == 4


class TestPeriod:
    def test_period_is_nonnegative(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS)
        assert (df["period"] >= 0).all()

    def test_period_increases_with_time(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS).reset_index(drop=True)
        # After sort by date, periods must be non-decreasing
        assert (df["period"].diff().dropna() >= 0).all()

    def test_earliest_row_period_zero_or_positive(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS).reset_index(drop=True)
        # Earliest match is Q4 2022 (Nov) → period = (2022-2022)*4 + (11-1)//3 = 3
        assert df.iloc[0]["period"] == 3

    def test_second_row_period_increments(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS).reset_index(drop=True)
        # Second earliest should be from Dec 2022 → same Q4 → period 3 also
        assert df.iloc[1]["period"] == 3

    def test_friendly_period(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS).reset_index(drop=True)
        fr = df[(df["competition"] == 10) & (df["home"] == "France")].iloc[0]
        # 2023-03-25 → (2023-2022)*4 + (3-1)//3 = 4 + 0 = 4
        assert fr["period"] == 4

    def test_brazil_arg_period(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS).reset_index(drop=True)
        bra = df[df["home"] == "Brazil"].iloc[0]
        # 2024-10-15 → (2024-2022)*4 + (10-1)//3 = 8 + 3 = 11
        assert bra["period"] == 11


class TestCutoff:
    def test_cutoff_drops_later_matches(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS, cutoff="2024-01-01")
        assert all(df["date"] < pd.Timestamp("2024-01-01", tz="UTC"))

    def test_cutoff_keeps_earlier_matches(self):
        df = assemble_internationals(_CLIENT, _COMP_SEASONS, cutoff="2024-01-01")
        # WC and one Friendly should remain; 2024 qual is cut
        assert len(df) == 3

    def test_cutoff_none_keeps_all(self):
        df_all = assemble_internationals(_CLIENT, _COMP_SEASONS)
        df_none = assemble_internationals(_CLIENT, _COMP_SEASONS, cutoff=None)
        assert len(df_all) == len(df_none)


def test_assemble_filters_youth_bteams_and_non_fifa():
    """Youth/B teams and friendlies-only non-FIFA entities are excluded."""

    class StubClient:
        def __init__(self, by_key):
            self.by_key = by_key

        def get_response(self, endpoint, params=None):
            return self.by_key.get((params["league"], params["season"]), [])

    def fx(h, a, comp):
        return {
            "fixture": {"date": "2024-03-01T00:00:00+00:00", "status": {"short": "FT"}},
            "league": {"id": comp, "season": 2024},
            "teams": {"home": {"name": h}, "away": {"name": a}},
            "goals": {"home": 1, "away": 0},
        }

    from cupcast.v2.model.matches import assemble_internationals

    fixtures = {
        (10, 2024): [
            fx("Spain", "Germany", 10),       # senior friendly, both official -> kept
            fx("Spain U21", "France", 10),     # youth -> dropped (senior filter)
            fx("Spain", "FC Urartu", 10),      # club, friendlies-only -> dropped (official filter)
            fx("Spain", "Catalonia", 10),      # non-FIFA, friendlies-only -> dropped
        ],
        (5, 2024): [  # Nations League (official) establishes Spain/Germany/France
            fx("Spain", "France", 5),
            fx("Germany", "France", 5),
        ],
    }
    m = assemble_internationals(StubClient(fixtures), [(10, 2024), (5, 2024)])
    names = set(m["home"]) | set(m["away"])
    assert "Spain U21" not in names
    assert "FC Urartu" not in names
    assert "Catalonia" not in names
    assert {"Spain", "Germany", "France"} <= names
    assert len(m) == 3
