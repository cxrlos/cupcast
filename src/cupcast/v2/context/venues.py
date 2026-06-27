from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    city: str
    country: str
    lat: float
    lon: float
    elevation_m: int
    host_nation: str
    hosts_through_round: str


_CANONICAL: dict[str, Venue] = {
    # Mexico
    "Mexico City": Venue("Mexico City", "Mexico", 19.30, -99.15, 2240, "Mexico", "R16"),
    "Zapopan": Venue("Zapopan", "Mexico", 20.68, -103.46, 1560, "Mexico", "R16"),
    "Monterrey": Venue("Monterrey", "Mexico", 25.67, -100.31, 540, "Mexico", "R16"),
    # Canada
    "Toronto": Venue("Toronto", "Canada", 43.63, -79.42, 76, "Canada", "R16"),
    "Vancouver": Venue("Vancouver", "Canada", 49.28, -123.11, 0, "Canada", "R16"),
    # USA
    "Boston": Venue("Boston", "USA", 42.09, -71.26, 90, "USA", "Final"),
    "Houston": Venue("Houston", "USA", 29.68, -95.41, 15, "USA", "Final"),
    "Philadelphia": Venue("Philadelphia", "USA", 39.90, -75.17, 12, "USA", "Final"),
    "Atlanta": Venue("Atlanta", "USA", 33.75, -84.40, 320, "USA", "Final"),
    "Inglewood": Venue("Inglewood", "USA", 33.95, -118.34, 30, "USA", "Final"),
    "Santa Clara": Venue("Santa Clara", "USA", 37.40, -121.97, 5, "USA", "Final"),
    "East Rutherford": Venue("East Rutherford", "USA", 40.81, -74.07, 5, "USA", "Final"),
    "Seattle": Venue("Seattle", "USA", 47.59, -122.33, 5, "USA", "Final"),
    "Kansas City": Venue("Kansas City", "USA", 39.05, -94.48, 270, "USA", "Final"),
    "Arlington": Venue("Arlington", "USA", 32.75, -97.09, 150, "USA", "Final"),
    "Miami Gardens": Venue("Miami Gardens", "USA", 25.96, -80.24, 3, "USA", "Final"),
}

_ALIASES: dict[str, str] = {
    "Los Angeles": "Inglewood",
    "San Francisco Bay Area": "Santa Clara",
    "New York New Jersey": "East Rutherford",
    "Dallas": "Arlington",
    "Miami": "Miami Gardens",
}

VENUES: dict[str, Venue] = {
    **_CANONICAL,
    **{alias: _CANONICAL[canon] for alias, canon in _ALIASES.items()},
}


def venue_for(city: str) -> Venue | None:
    return VENUES.get(city)
