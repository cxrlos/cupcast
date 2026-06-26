from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v1.model.dixon_coles import DixonColesFit
from cupcast.v1.sim.monte_carlo import GROUP_LETTERS, TournamentDetails
from cupcast.v1.sim.worldcup2026 import (
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

STAGE_NAMES = {
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-finals",
    "sf": "Semi-finals",
    "third": "Third-place play-off",
    "final": "Final",
}


def confidence_score(p_home: float, p_draw: float, p_away: float) -> int:
    """1-5 confidence from the entropy of the W/D/L distribution."""
    probs = np.array([p_home, p_draw, p_away])
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum() / np.log(3))
    return int(np.clip(np.ceil((1 - entropy) * 7.5), 1, 5))


def predict_fixture(
    fit: DixonColesFit, home: str, away: str, venue_country: str | None = None
) -> dict:
    host_home = home in HOST_COUNTRIES and (venue_country is None or home == venue_country)
    host_away = away in HOST_COUNTRIES and venue_country == away
    matrix = fit.score_matrix(home, away, host_home=host_home, host_away=host_away)
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = 1 - p_home - p_draw
    modal = np.unravel_index(int(matrix.argmax()), matrix.shape)
    goals = np.arange(matrix.shape[0])
    return {
        "home": home,
        "away": away,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "modal_score": f"{modal[0]}-{modal[1]}",
        "xg_home": float(goals @ matrix.sum(axis=1)),
        "xg_away": float(goals @ matrix.sum(axis=0)),
        "confidence": confidence_score(p_home, p_draw, p_away),
    }


def _note(fit: DixonColesFit, prediction: dict) -> str:
    home, away = prediction["home"], prediction["away"]
    edge = (fit.attack[fit.teams.index(home)] + fit.defense[fit.teams.index(home)]) - (
        fit.attack[fit.teams.index(away)] + fit.defense[fit.teams.index(away)]
    )
    parts = []
    favorite = home if prediction["p_home"] >= prediction["p_away"] else away
    margin = abs(prediction["p_home"] - prediction["p_away"])
    if margin < 0.10:
        parts.append("Near coin-flip on ratings; group context will decide approach.")
    elif abs(edge) > 0.8:
        parts.append(f"{favorite} are a class above on rating ({edge:+.2f} net).")
    else:
        parts.append(f"{favorite} favored, but within one rating tier ({edge:+.2f} net).")
    if home in HOST_COUNTRIES:
        parts.append(f"{home} carry host advantage here.")
    if prediction["p_draw"] > 0.30:
        parts.append("Elevated draw risk; a cagey low-scoring game is the modal pattern.")
    return " ".join(parts)


def group_stage_predictions(fit: DixonColesFit) -> pd.DataFrame:
    rows = []
    for letter in GROUP_LETTERS:
        teams = GROUPS[letter]
        for i in range(4):
            for j in range(i + 1, 4):
                prediction = predict_fixture(fit, teams[i], teams[j])
                rows.append(
                    {"stage": "Group " + letter, **prediction, "note": _note(fit, prediction)}
                )
    return pd.DataFrame(rows)


def knockout_projections(fit: DixonColesFit, details: TournamentDetails) -> pd.DataFrame:
    """Modal matchup per knockout match with conditional advance probability."""
    n_sims = details.winners.shape[1]
    rows = []
    stages = (
        [("r32", m) for m in R32]
        + [("r16", m) for m in R16]
        + [("qf", m) for m in QUARTERFINALS]
        + [("sf", m) for m in SEMIFINALS]
        + [("third", THIRD_PLACE), ("final", FINAL)]
    )
    for stage_key, match_spec in stages:
        match_id, *_, venue = match_spec
        winner_ids = details.match_winner[match_id]
        loser_ids = details.match_loser[match_id]
        lo = np.minimum(winner_ids, loser_ids)
        hi = np.maximum(winner_ids, loser_ids)
        codes = lo * len(ALL_TEAMS) + hi
        top_code, count = np.unique(codes, return_counts=True)
        best = top_code[count.argmax()]
        share = count.max() / n_sims
        a, b = divmod(int(best), len(ALL_TEAMS))
        team_a, team_b = ALL_TEAMS[a], ALL_TEAMS[b]
        in_pair = codes == best
        p_a_advances = float((winner_ids[in_pair] == a).mean())
        prediction = predict_fixture(fit, team_a, team_b, venue)
        rows.append(
            {
                "stage": STAGE_NAMES[stage_key],
                "match": match_id,
                "modal_pairing": f"{team_a} vs {team_b}",
                "p_this_pairing": share,
                "p_first_advances": p_a_advances,
                **{k: v for k, v in prediction.items() if k not in ("home", "away")},
            }
        )
    return pd.DataFrame(rows)
