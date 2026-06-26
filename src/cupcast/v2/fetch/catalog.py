from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

INTERNATIONAL_COMPETITIONS: tuple[str, ...] = (
    "World Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "Euro Championship",
    "Copa America",
    "Africa Cup of Nations",
    "Asian Cup",
    "Gold Cup",
    "Friendlies",
)

# ISO-3 club_country code -> catalog country.name, for the leagues that supply
# World Cup squad players. Extend as squads dictate.
_COUNTRY_BY_CODE = {
    "ENG": "England", "ESP": "Spain", "ITA": "Italy", "GER": "Germany",
    "FRA": "France", "NED": "Netherlands", "POR": "Portugal", "BEL": "Belgium",
    "TUR": "Turkey", "USA": "USA", "MEX": "Mexico", "BRA": "Brazil",
    "ARG": "Argentina", "KSA": "Saudi-Arabia", "ENG2": "England",
    "SCO": "Scotland", "GRE": "Greece", "SUI": "Switzerland", "AUT": "Austria",
}


def _is_world(entry: dict) -> bool:
    return (entry.get("country") or {}).get("name") == "World"


def resolve_competitions(catalog: list[dict], names: Iterable[str]) -> list[dict]:
    wanted = [n.casefold() for n in names]
    out: list[dict] = []
    for needle in wanted:
        matches = [
            e for e in catalog if needle in e["league"]["name"].casefold()
        ]
        world = [e for e in matches if _is_world(e)]
        chosen = world or matches
        out.extend(chosen[:1] if world else chosen)
    # de-duplicate by league id, preserve order
    seen, unique = set(), []
    for e in out:
        lid = e["league"]["id"]
        if lid not in seen:
            seen.add(lid)
            unique.append(e)
    return unique


def club_leagues_from_squads(catalog: list[dict], squads: pd.DataFrame) -> list[dict]:
    countries = {
        _COUNTRY_BY_CODE.get(code)
        for code in squads["club_country"].dropna().unique()
    }
    countries.discard(None)
    out = []
    for country in sorted(countries):
        leagues = [
            e for e in catalog
            if (e.get("country") or {}).get("name") == country
            and e["league"].get("type") == "League"
        ]
        if leagues:
            out.append(min(leagues, key=lambda e: e["league"]["id"]))
    return out


def seasons_between(entry: dict, first: int, last: int) -> list[int]:
    return sorted(s["year"] for s in entry.get("seasons", []) if first <= s["year"] <= last)
