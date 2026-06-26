from __future__ import annotations

from cupcast.v1.fetch.api_football import ApiFootballClient


def load_leagues(client: ApiFootballClient) -> list[dict]:
    return client.get_response("leagues")


def find_league(catalog: list[dict], name: str) -> dict | None:
    matches = [e for e in catalog if e["league"]["name"].casefold() == name.casefold()]
    if not matches:
        return None
    # International competitions are filed under country "World"; prefer them over
    # same-named domestic leagues.
    world = [e for e in matches if (e.get("country") or {}).get("name") == "World"]
    return (world or matches)[0]


def search_leagues(catalog: list[dict], term: str) -> list[dict]:
    needle = term.casefold()
    return [e for e in catalog if needle in e["league"]["name"].casefold()]


def seasons_between(entry: dict, first: int, last: int) -> list[int]:
    return sorted(s["year"] for s in entry.get("seasons", []) if first <= s["year"] <= last)
