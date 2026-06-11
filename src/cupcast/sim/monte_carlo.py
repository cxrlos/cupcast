from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cupcast.model.dixon_coles import DixonColesFit
from cupcast.sim.bracket import allocate_thirds
from cupcast.sim.group_stage import composite_key, rank_group, sample_group_scores
from cupcast.sim.knockout import KnockoutSampler
from cupcast.sim.worldcup2026 import (
    ALL_TEAMS,
    FINAL,
    GROUPS,
    HOST_COUNTRIES,
    QUARTERFINALS,
    R16,
    R32,
    SEMIFINALS,
    THIRD_PLACE,
)

GROUP_LETTERS = tuple(GROUPS)


@dataclass
class TournamentDetails:
    table: pd.DataFrame
    winners: np.ndarray  # (12, n_sims) team ids of group winners
    runners: np.ndarray
    thirds: np.ndarray
    qualifies: np.ndarray  # (12, n_sims) bool, third qualified
    match_winner: dict[int, np.ndarray]
    match_loser: dict[int, np.ndarray]


def run_tournament(fit: DixonColesFit, n_sims: int = 50_000, seed: int = 2026) -> pd.DataFrame:
    return simulate_tournament(fit, n_sims, seed).table


def simulate_tournament(
    fit: DixonColesFit, n_sims: int = 50_000, seed: int = 2026
) -> TournamentDetails:
    rng = np.random.default_rng(seed)
    team_id = {team: i for i, team in enumerate(ALL_TEAMS)}
    n_teams = len(ALL_TEAMS)
    sims = np.arange(n_sims)

    winners = np.empty((12, n_sims), dtype=np.int64)
    runners = np.empty((12, n_sims), dtype=np.int64)
    thirds = np.empty((12, n_sims), dtype=np.int64)
    third_keys = np.empty((12, n_sims), dtype=np.float64)
    exp_points = np.zeros(n_teams)
    exp_gf = np.zeros(n_teams)
    exp_ga = np.zeros(n_teams)

    for g, letter in enumerate(GROUP_LETTERS):
        teams = GROUPS[letter]
        ids = np.array([team_id[t] for t in teams])
        hosts = tuple(t in HOST_COUNTRIES for t in teams)
        home_goals, away_goals = sample_group_scores(fit, teams, hosts, n_sims, rng)
        order, points, gd, gf = rank_group(home_goals, away_goals, rng)
        winners[g] = ids[order[0]]
        runners[g] = ids[order[1]]
        thirds[g] = ids[order[2]]
        third_keys[g] = (
            composite_key(points, gd, gf)[order[2], sims] + rng.random(n_sims) * 0.5
        )
        ga = gf - gd
        for slot, tid in enumerate(ids):
            exp_points[tid] = points[slot].mean()
            exp_gf[tid] = gf[slot].mean()
            exp_ga[tid] = ga[slot].mean()

    # Eight best thirds per simulation: points, GD, GF, then lots (jitter).
    qualifies = np.zeros((12, n_sims), dtype=bool)
    top8 = np.argsort(-third_keys, axis=0, kind="stable")[:8]
    np.put_along_axis(qualifies, top8, True, axis=0)

    masks = np.zeros(n_sims, dtype=np.int64)
    for g in range(12):
        masks |= qualifies[g].astype(np.int64) << g

    third_slots = [match for match, _, spec_b, _ in R32 if spec_b[0] == "T"]
    slot_team = {slot: np.empty(n_sims, dtype=np.int64) for slot in third_slots}
    for mask in np.unique(masks):
        in_mask = masks == mask
        qualified = frozenset(GROUP_LETTERS[g] for g in range(12) if (int(mask) >> g) & 1)
        for slot, letter in allocate_thirds(qualified).items():
            slot_team[slot][in_mask] = thirds[GROUP_LETTERS.index(letter)][in_mask]

    def resolve(spec: tuple, match: int) -> np.ndarray:
        kind, value = spec
        if kind == "W":
            return winners[GROUP_LETTERS.index(value)]
        if kind == "R":
            return runners[GROUP_LETTERS.index(value)]
        return slot_team[match]

    sampler = KnockoutSampler(fit, ALL_TEAMS)
    match_winner: dict[int, np.ndarray] = {}
    match_loser: dict[int, np.ndarray] = {}

    def play(match: int, side_a: np.ndarray, side_b: np.ndarray, venue: str) -> None:
        won = sampler.sample_winners(side_a, side_b, venue, rng)
        match_winner[match] = won
        match_loser[match] = np.where(won == side_a, side_b, side_a)

    for match, spec_a, spec_b, venue in R32:
        play(match, resolve(spec_a, match), resolve(spec_b, match), venue)
    for stage in (R16, QUARTERFINALS, SEMIFINALS):
        for match, feed_a, feed_b, venue in stage:
            play(match, match_winner[feed_a], match_winner[feed_b], venue)

    third_match, sf1, sf2, third_venue = THIRD_PLACE
    play(third_match, match_loser[sf1], match_loser[sf2], third_venue)
    final_match, f1, f2, final_venue = FINAL
    play(final_match, match_winner[f1], match_winner[f2], final_venue)

    def share(ids: np.ndarray, scale: int = 1) -> np.ndarray:
        return np.bincount(ids, minlength=n_teams) / (n_sims * scale)

    qualified_third_ids = np.where(qualifies, thirds, -1).ravel()
    qualified_third_ids = qualified_third_ids[qualified_third_ids >= 0]
    round_entrants = {
        "p_r16": [match_winner[m] for m, *_ in R32],
        "p_qf": [match_winner[m] for m, *_ in R16],
        "p_sf": [match_winner[m] for m, *_ in QUARTERFINALS],
        "p_final": [match_winner[m] for m, *_ in SEMIFINALS],
    }

    table = pd.DataFrame(
        {
            "team": ALL_TEAMS,
            "group": [letter for letter in GROUP_LETTERS for _ in range(4)],
            "p_group_win": share(winners.ravel()),
            "p_group_runner_up": share(runners.ravel()),
            "p_group_third": share(thirds.ravel()),
            "p_qualify": share(winners.ravel())
            + share(runners.ravel())
            + share(qualified_third_ids),
            **{
                column: share(np.concatenate(entrants))
                for column, entrants in round_entrants.items()
            },
            "p_champion": share(match_winner[final_match]),
            "p_runner_up": share(match_loser[final_match]),
            "p_podium_third": share(match_winner[third_match]),
            "exp_points": exp_points,
            "exp_gf": exp_gf,
            "exp_ga": exp_ga,
        }
    )
    return TournamentDetails(
        table=table.sort_values("p_champion", ascending=False, ignore_index=True),
        winners=winners,
        runners=runners,
        thirds=thirds,
        qualifies=qualifies,
        match_winner=match_winner,
        match_loser=match_loser,
    )
