from __future__ import annotations

import numpy as np

from cupcast.model.dixon_coles import DixonColesFit

# All-play-all pairings by slot index within a group; listing order carries no
# home meaning at a World Cup, host advantage is flagged per team.
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

MAX_GOALS = 10

# Composite ranking key: points, then goal difference, then goals for, exactly
# the first three FIFA group tiebreakers. Offsets keep each component from
# bleeding into the next (|gd| <= 60, gf <= 60 with the goal cap).
def composite_key(points: np.ndarray, gd: np.ndarray, gf: np.ndarray) -> np.ndarray:
    return points * 100_000 + (gd + 100) * 100 + gf


def sample_group_scores(
    fit: DixonColesFit,
    teams: tuple[str, str, str, str],
    hosts: tuple[bool, bool, bool, bool],
    n_sims: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    side = MAX_GOALS + 1
    home_goals = np.empty((len(PAIRS), n_sims), dtype=np.int64)
    away_goals = np.empty((len(PAIRS), n_sims), dtype=np.int64)
    for m, (i, j) in enumerate(PAIRS):
        matrix = fit.score_matrix(
            teams[i], teams[j], host_home=hosts[i], host_away=hosts[j], max_goals=MAX_GOALS
        )
        flat = rng.choice(side * side, size=n_sims, p=matrix.ravel())
        home_goals[m] = flat // side
        away_goals[m] = flat % side
    return home_goals, away_goals


def group_tables(
    home_goals: np.ndarray, away_goals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sims = home_goals.shape[1]
    points = np.zeros((4, n_sims), dtype=np.int64)
    gf = np.zeros((4, n_sims), dtype=np.int64)
    ga = np.zeros((4, n_sims), dtype=np.int64)
    for m, (i, j) in enumerate(PAIRS):
        hg, ag = home_goals[m], away_goals[m]
        points[i] += 3 * (hg > ag) + (hg == ag)
        points[j] += 3 * (ag > hg) + (hg == ag)
        gf[i] += hg
        ga[i] += ag
        gf[j] += ag
        ga[j] += hg
    return points, gf - ga, gf, ga


def _head_to_head_order(
    tied: list[int],
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    sim: int,
    rng: np.random.Generator,
) -> list[int]:
    points = dict.fromkeys(tied, 0)
    gd = dict.fromkeys(tied, 0)
    gf = dict.fromkeys(tied, 0)
    for m, (i, j) in enumerate(PAIRS):
        if i not in points or j not in points:
            continue
        hg, ag = int(home_goals[m, sim]), int(away_goals[m, sim])
        points[i] += 3 * (hg > ag) + (hg == ag)
        points[j] += 3 * (ag > hg) + (hg == ag)
        gd[i] += hg - ag
        gd[j] += ag - hg
        gf[i] += hg
        gf[j] += ag
    # Final tiebreaker is drawing of lots (fair play is not modelled; see the
    # methodology paper).
    return sorted(tied, key=lambda t: (-points[t], -gd[t], -gf[t], rng.random()))


def rank_group(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, gd, gf, _ = group_tables(home_goals, away_goals)
    keys = composite_key(points, gd, gf).astype(np.float64)
    jitter = rng.random(keys.shape) * 0.5  # drawing-of-lots stand-in
    order = np.argsort(-(keys + jitter), axis=0, kind="stable")

    # Rows where two teams share the exact composite key need the head-to-head
    # tiebreakers instead of plain lots.
    sorted_keys = np.take_along_axis(keys, order, axis=0)
    tied_rows = np.flatnonzero((np.diff(sorted_keys, axis=0) == 0).any(axis=0))
    for sim in tied_rows:
        sim_keys = keys[:, sim]
        ranking: list[int] = []
        for key_value in sorted(set(sim_keys), reverse=True):
            cluster = [int(t) for t in np.flatnonzero(sim_keys == key_value)]
            if len(cluster) == 1:
                ranking.extend(cluster)
            else:
                ranking.extend(_head_to_head_order(cluster, home_goals, away_goals, sim, rng))
        order[:, sim] = ranking
    return order, points, gd, gf
