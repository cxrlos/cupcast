from __future__ import annotations

from functools import cache

# Allowed source groups for each best-third slot in the round of 32, from the
# official bracket (match number -> groups whose third can fill the slot).
SLOT_ALLOWED: dict[int, str] = {
    74: "ABCDF",
    77: "CDFGH",
    79: "CEFHI",
    80: "EHIJK",
    81: "BEFIJ",
    82: "AEHIJ",
    85: "EFGIJ",
    87: "DEIJL",
}


@cache
def allocate_thirds(qualified: frozenset[str]) -> dict[int, str]:
    """Assign the eight qualified third-placed groups to bracket slots.

    Deterministic backtracking: most-constrained slot first (match number
    breaks ties), alphabetically smallest group first. Reproduces the worked
    example FIFA published from Annex C of the tournament regulations.
    """
    if len(qualified) != 8:
        raise ValueError(f"exactly 8 third-placed groups qualify, got {len(qualified)}")

    def solve(assignment: dict[int, str], available: frozenset[str]) -> dict[int, str] | None:
        if len(assignment) == len(SLOT_ALLOWED):
            return assignment
        open_slots = [s for s in SLOT_ALLOWED if s not in assignment]
        slot = min(open_slots, key=lambda s: (len(set(SLOT_ALLOWED[s]) & available), s))
        for group in sorted(set(SLOT_ALLOWED[slot]) & available):
            result = solve({**assignment, slot: group}, available - {group})
            if result is not None:
                return result
        return None

    solution = solve({}, frozenset(qualified))
    if solution is None:
        raise ValueError(f"no valid third-place allocation for {sorted(qualified)}")
    return solution
