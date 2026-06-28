"""TDD tests for cupcast.v2.sim.live_bracket."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cupcast.v2.sim.live_bracket import (
    resolve_live_r32,
    simulate_live_knockouts,
    validate_live_r32,
)
from cupcast.v2.sim.structure import R32

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_GROUPS = "ABCDEFGHIJKL"
_STUB_WINNERS = {g: f"W_{g}" for g in _GROUPS}
_STUB_RUNNERS = {g: f"R_{g}" for g in _GROUPS}

# T-slot anchors (spec_a team) and the chosen third placed in the fixture.
# Each T slot: anchor = W_{group from spec_a}, third = "T_{match_number}".
_T_SLOTS = {
    74: ("W_E", "T_74"),
    77: ("W_I", "T_77"),
    79: ("W_A", "T_79"),
    80: ("W_L", "T_80"),
    81: ("W_D", "T_81"),
    82: ("W_G", "T_82"),
    85: ("W_B", "T_85"),
    87: ("W_K", "T_87"),
}

# Build a full set of 16 R32 fixtures from known (teamA, teamB) pairs.
# W/R slots resolve directly; T slots use the anchor from _T_SLOTS.
def _build_expected_pairs() -> dict[int, tuple[str, str]]:
    pairs: dict[int, tuple[str, str]] = {}
    for match, spec_a, spec_b, _venue in R32:
        kind_a, val_a = spec_a
        team_a = _STUB_WINNERS[val_a] if kind_a == "W" else _STUB_RUNNERS[val_a]
        if spec_b[0] == "T":
            team_b = _T_SLOTS[match][1]
        else:
            kind_b, val_b = spec_b
            team_b = _STUB_WINNERS[val_b] if kind_b == "W" else _STUB_RUNNERS[val_b]
        pairs[match] = (team_a, team_b)
    return pairs


_EXPECTED_PAIRS = _build_expected_pairs()


def _make_fixture(team_a: str, team_b: str) -> dict:
    return {"teams": {"home": {"name": team_a}, "away": {"name": team_b}}}


def _build_fixtures(pairs: dict[int, tuple[str, str]]) -> list[dict]:
    return [_make_fixture(a, b) for a, b in pairs.values()]


_STUB_FIXTURES = _build_fixtures(_EXPECTED_PAIRS)


# ---------------------------------------------------------------------------
# resolve_live_r32
# ---------------------------------------------------------------------------


def test_resolve_wr_slots() -> None:
    """Pure W/R slots resolve to their standings entry."""
    resolved = resolve_live_r32(_STUB_WINNERS, _STUB_RUNNERS, _STUB_FIXTURES)
    # Slot 73: ("R","A"), ("R","B") → R_A, R_B
    assert resolved[73] == ("R_A", "R_B")
    # Slot 75: ("W","F"), ("R","C") → W_F, R_C
    assert resolved[75] == ("W_F", "R_C")
    # Slot 84: ("W","H"), ("R","J") → W_H, R_J
    assert resolved[84] == ("W_H", "R_J")


def test_resolve_t_slot_uses_actual_fixture_not_allocate_thirds() -> None:
    """Third-placed team must come from the real fixture, not allocate_thirds."""
    resolved = resolve_live_r32(_STUB_WINNERS, _STUB_RUNNERS, _STUB_FIXTURES)
    for match, (anchor, third) in _T_SLOTS.items():
        a, b = resolved[match]
        assert a == anchor, f"Slot {match}: expected anchor {anchor!r}, got {a!r}"
        assert b == third, (
            f"Slot {match}: expected third {third!r} from fixture, got {b!r}"
        )


def test_resolve_returns_all_16_slots() -> None:
    resolved = resolve_live_r32(_STUB_WINNERS, _STUB_RUNNERS, _STUB_FIXTURES)
    r32_match_numbers = {m for m, *_ in R32}
    assert set(resolved.keys()) == r32_match_numbers


def test_resolve_raises_on_missing_anchor() -> None:
    """If the anchor is not in any fixture, resolve must raise."""
    bad_winners = {**_STUB_WINNERS, "E": "GHOST_TEAM"}
    with pytest.raises(ValueError, match="GHOST_TEAM"):
        resolve_live_r32(bad_winners, _STUB_RUNNERS, _STUB_FIXTURES)


# ---------------------------------------------------------------------------
# validate_live_r32
# ---------------------------------------------------------------------------


def test_validate_ok_on_full_resolved() -> None:
    result = validate_live_r32(_EXPECTED_PAIRS, _STUB_FIXTURES)
    assert result["ok"] is True
    assert result["n_slots"] == 16
    assert result["n_distinct_teams"] == 32
    assert result["unmatched_fixtures"] == []
    assert result["issues"] == []


def test_validate_detects_wrong_pair() -> None:
    bad = {**_EXPECTED_PAIRS, 73: ("BOGUS_A", "BOGUS_B")}
    result = validate_live_r32(bad, _STUB_FIXTURES)
    assert result["ok"] is False
    assert any("73" in issue for issue in result["issues"])


def test_validate_detects_duplicate_teams() -> None:
    """Replacing one pair with a duplicate team triggers ok=False."""
    # Reuse T_74 on two slots → only 31 distinct teams
    bad = {**_EXPECTED_PAIRS, 73: ("T_74", "R_B")}
    result = validate_live_r32(bad, _STUB_FIXTURES)
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# simulate_live_knockouts
# ---------------------------------------------------------------------------

class _StubPosterior:
    """Fixed-rate stub; ignores team names and host flags."""

    rho = 0.0

    def rate(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
    ) -> tuple[float, float]:
        return 1.5, 1.5


_STUB_POST = _StubPosterior()
_N_SIMS = 400
_SEED = 42

# Resolved bracket with 32 distinct stub teams, one per R32 slot side.
_SIM_RESOLVED: dict[int, tuple[str, str]] = {
    m: (f"A{i}", f"B{i}") for i, (m, *_) in enumerate(R32)
}


@pytest.fixture(scope="module")
def sim_result() -> pd.DataFrame:
    return simulate_live_knockouts(
        _STUB_POST, _SIM_RESOLVED, n_sims=_N_SIMS, seed=_SEED
    )


def test_sim_returns_32_rows(sim_result) -> None:
    assert len(sim_result) == 32


def test_sim_has_required_columns(sim_result) -> None:
    expected = {
        "team", "p_r16", "p_qf", "p_sf", "p_final",
        "p_champion", "p_runner_up", "p_podium_third",
    }
    assert expected.issubset(set(sim_result.columns))


def test_sim_probs_in_unit_interval(sim_result) -> None:
    prob_cols = ["p_r16", "p_qf", "p_sf", "p_final", "p_champion", "p_runner_up", "p_podium_third"]
    for col in prob_cols:
        vals = sim_result[col].to_numpy()
        assert vals.min() >= -1e-9, f"{col} has value below 0"
        assert vals.max() <= 1.0 + 1e-9, f"{col} has value above 1"


def test_sim_p_champion_sums_to_one(sim_result) -> None:
    total = sim_result["p_champion"].sum()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_sim_round_monotonicity(sim_result) -> None:
    """p_r16 >= p_qf >= p_sf >= p_final >= p_champion per team."""
    df = sim_result
    tol = 1e-9
    pairs = [("p_r16", "p_qf"), ("p_qf", "p_sf"), ("p_sf", "p_final"), ("p_final", "p_champion")]
    for a, b in pairs:
        bad = df[df[a] < df[b] - tol]
        assert bad.empty, f"{a} < {b} for teams: {bad['team'].tolist()}"


def test_sim_sorted_by_p_champion(sim_result) -> None:
    champ = sim_result["p_champion"].to_numpy()
    assert (np.diff(champ) <= 1e-9).all()


def test_sim_reproducibility() -> None:
    a = simulate_live_knockouts(_STUB_POST, _SIM_RESOLVED, n_sims=_N_SIMS, seed=_SEED)
    b = simulate_live_knockouts(_STUB_POST, _SIM_RESOLVED, n_sims=_N_SIMS, seed=_SEED)
    np.testing.assert_array_equal(
        a["p_champion"].to_numpy(), b["p_champion"].to_numpy()
    )


def test_sim_all_32_teams_present(sim_result) -> None:
    expected_teams = {t for m, (a, b) in _SIM_RESOLVED.items() for t in (a, b)}
    assert set(sim_result["team"]) == expected_teams
