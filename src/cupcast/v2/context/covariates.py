from __future__ import annotations

import math
from datetime import UTC, datetime

from cupcast.v2.context.venues import Venue, venue_for

_R_KM = 6371.0


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return _R_KM * 2 * math.asin(math.sqrt(a))


def team_schedule(fixtures: list[dict], team: str) -> list[dict]:
    """WC2026 fixtures for *team* (home or away), sorted by fixture date."""
    matches = [
        f for f in fixtures
        if f["teams"]["home"]["name"] == team or f["teams"]["away"]["name"] == team
    ]
    matches.sort(key=lambda f: f["fixture"]["date"])
    return matches


def _parse_date(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fixture_city(fixture: dict) -> str:
    return fixture["fixture"]["venue"]["city"]


def match_context(fixtures: list[dict], team: str, fixture: dict) -> dict:
    """Raw (uncapped) context covariates for *team* in *fixture*.

    Returns a dict with: rest_days, travel_km, is_host, altitude_m, venue.
    rest_days is None for the team's first WC2026 match; travel_km is 0.0.
    """
    venue: Venue | None = venue_for(_fixture_city(fixture))

    schedule = team_schedule(fixtures, team)
    this_date = _parse_date(fixture["fixture"]["date"])

    # Find the immediately preceding fixture for this team
    prev: dict | None = None
    for f in schedule:
        f_date = _parse_date(f["fixture"]["date"])
        if f_date < this_date:
            prev = f
        else:
            break

    if prev is None:
        rest_days = None
        travel_km = 0.0
    else:
        prev_date = _parse_date(prev["fixture"]["date"])
        rest_days = (this_date - prev_date).days

        prev_venue = venue_for(_fixture_city(prev))
        if venue is not None and prev_venue is not None:
            travel_km = great_circle_km(
                prev_venue.lat, prev_venue.lon, venue.lat, venue.lon
            )
        else:
            travel_km = 0.0

    is_host = venue is not None and venue.host_nation == team
    altitude_m = venue.elevation_m if venue is not None else None

    return {
        "rest_days": rest_days,
        "travel_km": travel_km,
        "is_host": is_host,
        "altitude_m": altitude_m,
        "venue": venue,
    }
