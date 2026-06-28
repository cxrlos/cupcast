from __future__ import annotations

# Confirmed group draw after the March 2026 playoffs, per the official schedule
# (sources cited in docs/v1/tex/01-methodology, data section).
GROUPS: dict[str, tuple[str, str, str, str]] = {
    "A": ("Mexico", "South Africa", "South Korea", "Czech Republic"),
    "B": ("Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"),
    "C": ("Brazil", "Morocco", "Haiti", "Scotland"),
    "D": ("United States", "Paraguay", "Australia", "Turkey"),
    "E": ("Germany", "Curaçao", "Ivory Coast", "Ecuador"),
    "F": ("Netherlands", "Japan", "Sweden", "Tunisia"),
    "G": ("Belgium", "Egypt", "Iran", "New Zealand"),
    "H": ("Spain", "Cape Verde", "Saudi Arabia", "Uruguay"),
    "I": ("France", "Senegal", "Iraq", "Norway"),
    "J": ("Argentina", "Algeria", "Austria", "Jordan"),
    "K": ("Portugal", "DR Congo", "Uzbekistan", "Colombia"),
    "L": ("England", "Croatia", "Ghana", "Panama"),
}

HOST_COUNTRIES = ("Mexico", "United States", "Canada")

# Hosts play all three group matches inside their own country per the official
# schedule, so the group-stage host flag is purely team-based.

# Round of 32. Specs: ("W", group) winner, ("R", group) runner-up,
# ("T", allowed-groups) a best-third slot. Venue country drives knockout host
# advantage for the three hosts.
R32: list[tuple[int, tuple, tuple, str]] = [
    (73, ("R", "A"), ("R", "B"), "United States"),
    (74, ("W", "E"), ("T", "ABCDF"), "United States"),
    (75, ("W", "F"), ("R", "C"), "Mexico"),
    (76, ("W", "C"), ("R", "F"), "United States"),
    (77, ("W", "I"), ("T", "CDFGH"), "United States"),
    (78, ("R", "E"), ("R", "I"), "United States"),
    (79, ("W", "A"), ("T", "CEFHI"), "Mexico"),
    (80, ("W", "L"), ("T", "EHIJK"), "United States"),
    (81, ("W", "D"), ("T", "BEFIJ"), "United States"),
    (82, ("W", "G"), ("T", "AEHIJ"), "United States"),
    (83, ("R", "K"), ("R", "L"), "Canada"),
    (84, ("W", "H"), ("R", "J"), "United States"),
    (85, ("W", "B"), ("T", "EFGIJ"), "Canada"),
    (86, ("W", "J"), ("R", "H"), "United States"),
    (87, ("W", "K"), ("T", "DEIJL"), "United States"),
    (88, ("R", "D"), ("R", "G"), "United States"),
]

# Later rounds: (match, feeder match 1, feeder match 2, venue country).
R16 = [
    (89, 74, 77, "United States"),
    (90, 73, 75, "United States"),
    (91, 76, 78, "United States"),
    (92, 79, 80, "Mexico"),
    (93, 83, 84, "United States"),
    (94, 81, 82, "United States"),
    (95, 86, 88, "United States"),
    (96, 85, 87, "Canada"),
]
QUARTERFINALS = [
    (97, 89, 90, "United States"),
    (98, 93, 94, "United States"),
    (99, 91, 92, "United States"),
    (100, 95, 96, "United States"),
]
SEMIFINALS = [
    (101, 97, 98, "United States"),
    (102, 99, 100, "United States"),
]
FINAL = (104, 101, 102, "United States")
THIRD_PLACE = (103, 101, 102, "United States")  # losers of the semifinals

ALL_TEAMS: tuple[str, ...] = tuple(team for group in GROUPS.values() for team in group)
TEAM_GROUP: dict[str, str] = {
    team: letter for letter, group in GROUPS.items() for team in group
}
