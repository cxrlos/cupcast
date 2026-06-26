"""2026 FIFA World Cup tournament structure (API-Football team names)."""

from __future__ import annotations

from cupcast.v2.fetch.endpoints import fetch_fixtures

# Confirmed group draw after the March 2026 playoffs, per the official schedule.
# Team names match API-Football spellings used throughout the v2 model.
GROUPS: dict[str, tuple[str, str, str, str]] = {
    "A": ("Mexico", "South Africa", "South Korea", "Czechia"),
    "B": ("Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"),
    "C": ("Brazil", "Morocco", "Haiti", "Scotland"),
    "D": ("USA", "Paraguay", "Australia", "Türkiye"),
    "E": ("Germany", "Curaçao", "Ivory Coast", "Ecuador"),
    "F": ("Netherlands", "Japan", "Sweden", "Tunisia"),
    "G": ("Belgium", "Egypt", "Iran", "New Zealand"),
    "H": ("Spain", "Cape Verde Islands", "Saudi Arabia", "Uruguay"),
    "I": ("France", "Senegal", "Iraq", "Norway"),
    "J": ("Argentina", "Algeria", "Austria", "Jordan"),
    "K": ("Portugal", "Congo DR", "Uzbekistan", "Colombia"),
    "L": ("England", "Croatia", "Ghana", "Panama"),
}

HOST_COUNTRIES = ("Mexico", "USA", "Canada")

# Round of 32. Slot types: ("W", group) winner, ("R", group) runner-up,
# ("T", allowed-groups) best-third placed team. Venue country drives knockout
# host advantage for the three hosts.
R32: list[tuple[int, tuple, tuple, str]] = [
    (73, ("R", "A"), ("R", "B"), "USA"),
    (74, ("W", "E"), ("T", "ABCDF"), "USA"),
    (75, ("W", "F"), ("R", "C"), "Mexico"),
    (76, ("W", "C"), ("R", "F"), "USA"),
    (77, ("W", "I"), ("T", "CDFGH"), "USA"),
    (78, ("R", "E"), ("R", "I"), "USA"),
    (79, ("W", "A"), ("T", "CEFHI"), "Mexico"),
    (80, ("W", "L"), ("T", "EHIJK"), "USA"),
    (81, ("W", "D"), ("T", "BEFIJ"), "USA"),
    (82, ("W", "G"), ("T", "AEHIJ"), "USA"),
    (83, ("R", "K"), ("R", "L"), "Canada"),
    (84, ("W", "H"), ("R", "J"), "USA"),
    (85, ("W", "B"), ("T", "EFGIJ"), "Canada"),
    (86, ("W", "J"), ("R", "H"), "USA"),
    (87, ("W", "K"), ("T", "DEIJL"), "USA"),
    (88, ("R", "D"), ("R", "G"), "USA"),
]

R16 = [
    (89, 74, 77, "USA"),
    (90, 73, 75, "USA"),
    (91, 76, 78, "USA"),
    (92, 79, 80, "Mexico"),
    (93, 83, 84, "USA"),
    (94, 81, 82, "USA"),
    (95, 86, 88, "USA"),
    (96, 85, 87, "Canada"),
]

QUARTERFINALS = [
    (97, 89, 90, "USA"),
    (98, 93, 94, "USA"),
    (99, 91, 92, "USA"),
    (100, 95, 96, "USA"),
]

SEMIFINALS = [
    (101, 97, 98, "USA"),
    (102, 99, 100, "USA"),
]

FINAL = (104, 101, 102, "USA")
THIRD_PLACE = (103, 101, 102, "USA")

ALL_TEAMS: tuple[str, ...] = tuple(team for group in GROUPS.values() for team in group)
TEAM_GROUP: dict[str, str] = {
    team: letter for letter, group in GROUPS.items() for team in group
}
GROUP_LETTERS: tuple[str, ...] = tuple(GROUPS.keys())


def derive_groups_from_fixtures(client) -> list[frozenset[str]]:
    """Cluster WC2026 group-stage fixtures into connected components.

    Fetches league 1 / season 2026 from the client, keeps only fixtures
    whose round name contains ``"Group Stage"``, builds an undirected graph of
    teams that play each other, and returns one frozenset per connected component.
    """
    fixtures = fetch_fixtures(client, 1, 2026)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for fixture in fixtures:
        league = fixture.get("league") or {}
        if "Group Stage" not in (league.get("round") or ""):
            continue
        teams = fixture.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if home and away:
            union(home, away)

    components: dict[str, set[str]] = {}
    for team in list(parent):
        root = find(team)
        components.setdefault(root, set()).add(team)

    return [frozenset(c) for c in components.values()]
