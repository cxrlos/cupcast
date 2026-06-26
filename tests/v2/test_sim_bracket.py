from __future__ import annotations

import pytest

from cupcast.v2.sim.bracket import SLOT_ALLOWED, allocate_thirds


def test_valid_allocation_all_slots_filled():
    result = allocate_thirds(frozenset("ABCDEFGH"))
    assert set(result.keys()) == set(SLOT_ALLOWED.keys())


def test_valid_allocation_each_group_in_allowed_set():
    result = allocate_thirds(frozenset("ABCDEFGH"))
    for slot, group in result.items():
        assert group in SLOT_ALLOWED[slot]


def test_valid_allocation_groups_are_distinct():
    result = allocate_thirds(frozenset("ABCDEFGH"))
    assert len(set(result.values())) == 8


def test_published_annex_c_example():
    # FIFA's worked example: thirds of groups E–L qualify.
    result = allocate_thirds(frozenset("EFGHIJKL"))
    assert result == {79: "E", 85: "J", 81: "I", 74: "F", 82: "H", 77: "G", 87: "L", 80: "K"}


def test_raises_on_fewer_than_eight_groups():
    with pytest.raises(ValueError, match="8"):
        allocate_thirds(frozenset("ABCDEFG"))


def test_raises_on_more_than_eight_groups():
    with pytest.raises(ValueError, match="8"):
        allocate_thirds(frozenset("ABCDEFGHI"))


def test_raises_on_empty():
    with pytest.raises(ValueError, match="8"):
        allocate_thirds(frozenset())


def test_deterministic_same_input_same_output():
    a = allocate_thirds(frozenset("ABCDEFGH"))
    b = allocate_thirds(frozenset("ABCDEFGH"))
    assert a == b


def test_order_independent_frozenset_input():
    # frozenset has no ordering; ensure different construction routes agree.
    groups_1 = frozenset(["A", "B", "C", "D", "E", "F", "G", "H"])
    groups_2 = frozenset({"H", "G", "F", "E", "D", "C", "B", "A"})
    assert allocate_thirds(groups_1) == allocate_thirds(groups_2)
