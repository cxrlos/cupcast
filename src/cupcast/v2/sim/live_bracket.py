"""Conditioned knockout simulator seeded from the actual R32 draw."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v2.sim.knockout import KnockoutSampler
from cupcast.v2.sim.structure import (
    FINAL,
    QUARTERFINALS,
    R16,
    R32,
    SEMIFINALS,
    THIRD_PLACE,
)


def _fixture_opponent(fixture: dict, anchor: str) -> str | None:
    """Return the opponent of anchor in fixture, or None if anchor not present."""
    teams = fixture.get("teams") or {}
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")
    if anchor == home:
        return away or None
    if anchor == away:
        return home or None
    return None


def _fixture_team_set(fixture: dict) -> frozenset[str]:
    teams = fixture.get("teams") or {}
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")
    return frozenset({home, away}) if home and away else frozenset()


def resolve_live_r32(
    winners: dict[str, str],
    runners: dict[str, str],
    r32_fixtures: list[dict],
) -> dict[int, tuple[str, str]]:
    """Map each R32 template slot to its actual (teamA, teamB) from real results.

    Third-placed slots are resolved by locating the anchor (W or R) team in the
    provided fixture list and taking its opponent, bypassing allocate_thirds.
    """

    def resolve_wr(spec: tuple) -> str:
        kind, value = spec
        if kind == "W":
            return winners[value]
        return runners[value]

    def resolve_t(anchor: str) -> str:
        for fixture in r32_fixtures:
            opponent = _fixture_opponent(fixture, anchor)
            if opponent is not None:
                return opponent
        raise ValueError(f"No R32 fixture found containing anchor '{anchor}'")

    resolved: dict[int, tuple[str, str]] = {}
    for match, spec_a, spec_b, _venue in R32:
        team_a = resolve_wr(spec_a)
        team_b = resolve_t(team_a) if spec_b[0] == "T" else resolve_wr(spec_b)
        resolved[match] = (team_a, team_b)
    return resolved


def validate_live_r32(
    resolved: dict[int, tuple[str, str]],
    r32_fixtures: list[dict],
) -> dict:
    """Validate that resolved slots cover the actual R32 fixtures exactly.

    Returns a report dict with keys: ok, n_slots, n_distinct_teams,
    unmatched_fixtures, issues.
    """
    issues: list[str] = []
    matched_indices: set[int] = set()

    for match, (a, b) in resolved.items():
        target = frozenset({a, b})
        found_idx: int | None = None
        for idx, fixture in enumerate(r32_fixtures):
            if _fixture_team_set(fixture) == target:
                found_idx = idx
                break
        if found_idx is None:
            issues.append(f"Slot {match}: no fixture matches {{{a!r}, {b!r}}}")
        elif found_idx in matched_indices:
            issues.append(f"Slot {match}: fixture {found_idx} already consumed")
        else:
            matched_indices.add(found_idx)

    unmatched = [f for i, f in enumerate(r32_fixtures) if i not in matched_indices]

    all_teams = [t for a, b in resolved.values() for t in (a, b)]
    n_distinct = len(set(all_teams))
    if n_distinct != 32:
        issues.append(f"Expected 32 distinct teams, got {n_distinct}")

    return {
        "ok": not issues,
        "n_slots": len(resolved),
        "n_distinct_teams": n_distinct,
        "unmatched_fixtures": unmatched,
        "issues": issues,
    }


def simulate_live_knockouts(
    posterior,
    resolved_r32: dict[int, tuple[str, str]],
    gk_z: dict | None = None,
    n_sims: int = 50_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate the knockout bracket forward from the actual R32 draw.

    Returns one row per team with round-reach probabilities, sorted by
    p_champion descending.
    """
    rng = np.random.default_rng(seed)

    # Build deterministic team index by first-appearance order (match number asc).
    seen: dict[str, int] = {}
    for m in sorted(resolved_r32):
        for t in resolved_r32[m]:
            if t not in seen:
                seen[t] = len(seen)
    team_names = tuple(seen)
    team_id = seen
    n_teams = len(team_names)

    r32_venue: dict[int, str] = {m: v for m, *_, v in R32}

    sampler = KnockoutSampler(posterior, team_names, gk_z)
    match_winner: dict[int, np.ndarray] = {}
    match_loser: dict[int, np.ndarray] = {}

    def play(match: int, side_a: np.ndarray, side_b: np.ndarray, venue: str) -> None:
        won = sampler.sample_winners(side_a, side_b, venue, rng)
        match_winner[match] = won
        match_loser[match] = np.where(won == side_a, side_b, side_a)

    for m in sorted(resolved_r32):
        a, b = resolved_r32[m]
        side_a = np.full(n_sims, team_id[a], dtype=np.int64)
        side_b = np.full(n_sims, team_id[b], dtype=np.int64)
        play(m, side_a, side_b, r32_venue[m])

    for stage in (R16, QUARTERFINALS, SEMIFINALS):
        for match, feed_a, feed_b, venue in stage:
            play(match, match_winner[feed_a], match_winner[feed_b], venue)

    third_match, sf1, sf2, third_venue = THIRD_PLACE
    play(third_match, match_loser[sf1], match_loser[sf2], third_venue)
    final_match, f1, f2, final_venue = FINAL
    play(final_match, match_winner[f1], match_winner[f2], final_venue)

    def share(ids: np.ndarray) -> np.ndarray:
        return np.bincount(ids, minlength=n_teams) / n_sims

    round_entrants = {
        "p_r16": [match_winner[m] for m, *_ in R32],
        "p_qf": [match_winner[m] for m, *_ in R16],
        "p_sf": [match_winner[m] for m, *_ in QUARTERFINALS],
        "p_final": [match_winner[m] for m, *_ in SEMIFINALS],
    }

    return pd.DataFrame(
        {
            "team": team_names,
            **{
                col: share(np.concatenate(entrants))
                for col, entrants in round_entrants.items()
            },
            "p_champion": share(match_winner[final_match]),
            "p_runner_up": share(match_loser[final_match]),
            "p_podium_third": share(match_winner[third_match]),
        }
    ).sort_values("p_champion", ascending=False, ignore_index=True)
