"""Tests for sim.structure (Task 1 TDD)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_client(
    fixture_pairs: list[tuple[str, str]], round_prefix: str = "Group Stage"
) -> object:
    fixtures = [
        {
            "league": {"round": f"{round_prefix} - {i + 1}"},
            "teams": {"home": {"name": home}, "away": {"name": away}},
        }
        for i, (home, away) in enumerate(fixture_pairs)
    ]

    class _Stub:
        def get_response(self, endpoint, params):
            return fixtures

    return _Stub()


# ---------------------------------------------------------------------------
# GROUPS invariants
# ---------------------------------------------------------------------------


def test_groups_has_12_groups():
    from cupcast.v2.sim.structure import GROUPS

    assert len(GROUPS) == 12


def test_groups_has_48_unique_teams():
    from cupcast.v2.sim.structure import GROUPS

    all_teams = [t for group in GROUPS.values() for t in group]
    assert len(all_teams) == 48
    assert len(set(all_teams)) == 48


def test_groups_letters_a_through_l():
    from cupcast.v2.sim.structure import GROUPS

    assert set(GROUPS.keys()) == set("ABCDEFGHIJKL")


def test_host_countries_subset_of_teams():
    from cupcast.v2.sim.structure import ALL_TEAMS, HOST_COUNTRIES

    for h in HOST_COUNTRIES:
        assert h in ALL_TEAMS, f"{h!r} not in ALL_TEAMS"


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------


def test_all_teams_length():
    from cupcast.v2.sim.structure import ALL_TEAMS

    assert len(ALL_TEAMS) == 48


def test_team_group_covers_all_teams():
    from cupcast.v2.sim.structure import ALL_TEAMS, TEAM_GROUP

    assert set(TEAM_GROUP.keys()) == set(ALL_TEAMS)


def test_team_group_values_are_valid_letters():
    from cupcast.v2.sim.structure import GROUP_LETTERS, TEAM_GROUP

    for letter in TEAM_GROUP.values():
        assert letter in GROUP_LETTERS


def test_group_letters_ordered():
    from cupcast.v2.sim.structure import GROUP_LETTERS

    assert tuple("ABCDEFGHIJKL") == GROUP_LETTERS


# ---------------------------------------------------------------------------
# R32 bracket
# ---------------------------------------------------------------------------


def test_r32_has_16_matches():
    from cupcast.v2.sim.structure import R32

    assert len(R32) == 16


def test_r32_slots_reference_valid_groups():
    from cupcast.v2.sim.structure import GROUP_LETTERS, R32

    valid = set(GROUP_LETTERS)
    for _, slot1, slot2, _ in R32:
        for slot in (slot1, slot2):
            kind = slot[0]
            assert kind in ("W", "R", "T"), f"Unknown slot kind: {kind!r}"
            if kind in ("W", "R"):
                assert slot[1] in valid, f"Invalid group letter {slot[1]!r} in {slot}"
            else:
                for ch in slot[1]:
                    assert ch in valid, f"Invalid group {ch!r} in T-slot {slot}"


def test_r32_match_numbers_unique():
    from cupcast.v2.sim.structure import R32

    nums = [m for m, _, _, _ in R32]
    assert len(nums) == len(set(nums))


# ---------------------------------------------------------------------------
# derive_groups_from_fixtures
# ---------------------------------------------------------------------------


def test_derive_groups_two_disjoint_groups():
    """Two isolated 4-team groups yield two frozensets."""
    from cupcast.v2.sim.structure import derive_groups_from_fixtures

    g1 = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    g2 = [("E", "F"), ("E", "G"), ("E", "H"), ("F", "G"), ("F", "H"), ("G", "H")]
    stub = _make_stub_client(g1 + g2)

    result = derive_groups_from_fixtures(stub)
    assert len(result) == 2
    assert frozenset(["A", "B", "C", "D"]) in result
    assert frozenset(["E", "F", "G", "H"]) in result


def test_derive_groups_filters_non_group_stage():
    """Fixtures in other rounds (e.g. Round of 32) are ignored."""
    from cupcast.v2.sim.structure import derive_groups_from_fixtures

    def _fix(round_name, home, away):
        return {
            "league": {"round": round_name},
            "teams": {"home": {"name": home}, "away": {"name": away}},
        }

    fixtures = [
        _fix("Group Stage - 1", "A", "B"),
        _fix("Round of 32", "X", "Y"),
        _fix("Group Stage - 2", "A", "C"),
        _fix("Group Stage - 3", "B", "C"),
    ]

    class _Stub:
        def get_response(self, endpoint, params):
            return fixtures

    result = derive_groups_from_fixtures(_Stub())
    assert len(result) == 1
    assert frozenset(["A", "B", "C"]) in result


def test_derive_groups_three_groups():
    """Three isolated 4-team groups yield three frozensets."""
    from cupcast.v2.sim.structure import derive_groups_from_fixtures

    def pairs(teams):
        return [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ]

    g1, g2, g3 = ["P", "Q", "R", "S"], ["T", "U", "V", "W"], ["X", "Y", "Z", "Ω"]
    stub = _make_stub_client(pairs(g1) + pairs(g2) + pairs(g3))

    result = derive_groups_from_fixtures(stub)
    assert len(result) == 3
    assert frozenset(g1) in result
    assert frozenset(g2) in result
    assert frozenset(g3) in result
