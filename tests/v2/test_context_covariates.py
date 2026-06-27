from __future__ import annotations

import pytest

from cupcast.v2.context.covariates import great_circle_km, match_context, team_schedule
from cupcast.v2.context.venues import venue_for

# ---------------------------------------------------------------------------
# Helpers: in-memory fixture builders
# ---------------------------------------------------------------------------

def _fixture(fid: int, date: str, city: str, home: str, away: str) -> dict:
    return {
        "fixture": {
            "id": fid,
            "date": date,
            "venue": {"name": "Stadium", "city": city},
            "status": {"short": "NS"},
        },
        "teams": {
            "home": {"name": home},
            "away": {"name": away},
        },
        "league": {"id": 1, "season": 2026, "round": "Group Stage - 1"},
    }


_MEXICO_MATCH_1 = _fixture(1, "2026-06-11T20:00:00+00:00", "Mexico City", "Mexico", "Germany")
_MEXICO_MATCH_2 = _fixture(2, "2026-06-16T20:00:00+00:00", "Monterrey", "Mexico", "Brazil")
_MEXICO_US_MATCH = _fixture(3, "2026-06-21T20:00:00+00:00", "Houston", "USA", "Mexico")

_MEXICO_FIXTURES = [_MEXICO_MATCH_2, _MEXICO_MATCH_1, _MEXICO_US_MATCH]  # deliberately unsorted


# ---------------------------------------------------------------------------
# great_circle_km
# ---------------------------------------------------------------------------

class TestGreatCircleKm:
    def test_mexico_city_to_monterrey(self):
        # Mexico City 19.30, -99.15  →  Monterrey 25.67, -100.31  ≈ 700 km
        dist = great_circle_km(19.30, -99.15, 25.67, -100.31)
        assert abs(dist - 700) < 80

    def test_same_point_is_zero(self):
        assert great_circle_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        d1 = great_circle_km(19.30, -99.15, 25.67, -100.31)
        d2 = great_circle_km(25.67, -100.31, 19.30, -99.15)
        assert d1 == pytest.approx(d2, rel=1e-9)


# ---------------------------------------------------------------------------
# venue_for
# ---------------------------------------------------------------------------

class TestVenueFor:
    def test_canonical_city(self):
        v = venue_for("Mexico City")
        assert v is not None
        assert v.elevation_m == 2240
        assert v.host_nation == "Mexico"

    def test_alias_dallas_resolves_to_arlington(self):
        v = venue_for("Dallas")
        assert v is not None
        assert v.city == "Arlington"

    def test_alias_and_canonical_same_object(self):
        assert venue_for("Dallas") is venue_for("Arlington")

    def test_unknown_city_returns_none(self):
        assert venue_for("Narnia") is None

    def test_san_francisco_alias(self):
        v = venue_for("San Francisco Bay Area")
        assert v is not None
        assert v.city == "Santa Clara"

    def test_los_angeles_alias(self):
        v = venue_for("Los Angeles")
        assert v is not None
        assert v.city == "Inglewood"

    def test_new_york_alias(self):
        v = venue_for("New York New Jersey")
        assert v is not None
        assert v.city == "East Rutherford"

    def test_miami_alias(self):
        v = venue_for("Miami")
        assert v is not None
        assert v.city == "Miami Gardens"

    def test_canada_venue(self):
        v = venue_for("Toronto")
        assert v is not None
        assert v.host_nation == "Canada"
        assert v.hosts_through_round == "R16"

    def test_usa_venue_hosts_to_final(self):
        v = venue_for("East Rutherford")
        assert v is not None
        assert v.host_nation == "USA"
        assert v.hosts_through_round == "Final"


# ---------------------------------------------------------------------------
# team_schedule
# ---------------------------------------------------------------------------

class TestTeamSchedule:
    def test_returns_only_team_fixtures(self):
        sched = team_schedule(_MEXICO_FIXTURES, "Mexico")
        assert len(sched) == 3

    def test_sorted_by_date(self):
        sched = team_schedule(_MEXICO_FIXTURES, "Mexico")
        dates = [f["fixture"]["date"] for f in sched]
        assert dates == sorted(dates)

    def test_unrelated_team_empty(self):
        assert team_schedule(_MEXICO_FIXTURES, "Argentina") == []


# ---------------------------------------------------------------------------
# match_context
# ---------------------------------------------------------------------------

class TestMatchContext:
    def test_first_match_travel_zero(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_1)
        assert ctx["travel_km"] == pytest.approx(0.0)

    def test_first_match_default_rest(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_1)
        # default rest for first match is None or a large value; we test it is None
        assert ctx["rest_days"] is None

    def test_second_match_positive_rest_days(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_2)
        assert ctx["rest_days"] == 5  # Jun 11 → Jun 16

    def test_second_match_positive_travel_km(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_2)
        # Mexico City → Monterrey
        assert ctx["travel_km"] > 0.0
        assert abs(ctx["travel_km"] - 700) < 80

    def test_is_host_true_at_mexican_venue(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_1)
        assert ctx["is_host"] is True

    def test_is_host_false_at_us_venue(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_US_MATCH)
        assert ctx["is_host"] is False

    def test_altitude_m_correct(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_1)
        assert ctx["altitude_m"] == 2240

    def test_venue_returned(self):
        ctx = match_context(_MEXICO_FIXTURES, "Mexico", _MEXICO_MATCH_1)
        assert ctx["venue"] is not None
        assert ctx["venue"].city == "Mexico City"

    def test_unknown_venue_city(self):
        fx = _fixture(99, "2026-06-11T20:00:00+00:00", "Narnia", "Mexico", "Germany")
        ctx = match_context([fx], "Mexico", fx)
        assert ctx["venue"] is None
        assert ctx["altitude_m"] is None
        assert ctx["is_host"] is False
