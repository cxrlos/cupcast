from __future__ import annotations

_FINAL_STATUSES = {"FT", "AET", "PEN"}


def fetch_fixtures(client, league: int, season: int) -> list[dict]:
    return client.get_response("fixtures", {"league": league, "season": season})


def fetch_lineups(client, fixture: int) -> list[dict]:
    return client.get_response("fixtures/lineups", {"fixture": fixture})


def fetch_players(client, league: int, season: int) -> list[dict]:
    return client.get_response("players", {"league": league, "season": season})


def fetch_injuries(client, league: int, season: int) -> list[dict]:
    return client.get_response("injuries", {"league": league, "season": season})


def fixture_ids(fixtures: list[dict]) -> list[int]:
    ids = []
    for f in fixtures:
        fixture = f.get("fixture", {}) or {}
        status = (fixture.get("status", {}) or {}).get("short")
        fid = fixture.get("id")
        if status in _FINAL_STATUSES and fid is not None:
            ids.append(fid)
    return ids
