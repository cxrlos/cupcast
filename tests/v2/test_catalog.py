import pandas as pd

from cupcast.v2.fetch.catalog import (
    club_leagues_from_squads,
    resolve_competitions,
    seasons_between,
)

CATALOG = [
    {"league": {"id": 1, "name": "World Cup", "type": "Cup"},
     "country": {"name": "World", "code": None},
     "seasons": [{"year": 2018}, {"year": 2022}, {"year": 2026}]},
    {"league": {"id": 10, "name": "Friendlies", "type": "Cup"},
     "country": {"name": "World", "code": None},
     "seasons": [{"year": 2024}, {"year": 2025}]},
    {"league": {"id": 39, "name": "Premier League", "type": "League"},
     "country": {"name": "England", "code": "GB-ENG"},
     "seasons": [{"year": 2024}, {"year": 2025}]},
    {"league": {"id": 140, "name": "La Liga", "type": "League"},
     "country": {"name": "Spain", "code": "ES"},
     "seasons": [{"year": 2024}, {"year": 2025}]},
    {"league": {"id": 999, "name": "World Cup", "type": "Cup"},
     "country": {"name": "Brazil", "code": "BR"},
     "seasons": [{"year": 2022}]},
]


def test_resolve_competitions_prefers_world():
    # CATALOG has TWO "World Cup" entries: id 1 (World) and id 999 (Brazil).
    # The resolver must pick the World one and drop the domestic namesake.
    got = resolve_competitions(CATALOG, ["World Cup", "Friendlies"])
    ids = sorted(e["league"]["id"] for e in got)
    assert ids == [1, 10]
    assert 999 not in ids


def test_seasons_between():
    wc = CATALOG[0]
    assert seasons_between(wc, 2018, 2022) == [2018, 2022]


def test_club_leagues_from_squads():
    squads = pd.DataFrame({"club_country": ["ENG", "ENG", "ESP", "FRA"]})
    leagues = club_leagues_from_squads(CATALOG, squads)
    names = sorted(e["league"]["name"] for e in leagues)
    # FRA has no league in the catalog -> silently skipped; ENG + ESP resolved.
    assert names == ["La Liga", "Premier League"]
