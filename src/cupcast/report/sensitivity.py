from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cupcast.features.expected_minutes import expected_minutes_per_match
from cupcast.features.squad_strength import normalize_name, squad_composites
from cupcast.sim.monte_carlo import GROUP_LETTERS, TournamentDetails, simulate_tournament
from cupcast.sim.worldcup2026 import ALL_TEAMS, TEAM_GROUP

SQUADS_CSV = Path("data/processed/squads.csv")

# Star-player perturbations required by the project brief, plus symmetric
# checks on the other top contenders. minutes_factor 0 = withdrawn.
@dataclass(frozen=True)
class Scenario:
    name: str
    team: str
    player: str
    minutes_factor: float


SCENARIOS = (
    Scenario("Rodri out", "Spain", "Rodri", 0.0),
    Scenario("Lamine Yamal out", "Spain", "Lamine Yamal", 0.0),
    Scenario("Bellingham at 70 percent", "England", "Jude Bellingham", 0.7),
    Scenario("Messi limited to ~60 minutes", "Argentina", "Lionel Messi", 0.67),
    Scenario("Mbappé out", "France", "Kylian Mbappé", 0.0),
    Scenario("Vinícius Júnior out", "Brazil", "Vinícius Júnior", 0.0),
)

CONDITIONAL_TEAMS = (
    "Spain",
    "Argentina",
    "England",
    "Japan",
    "France",
    "Brazil",
    "United States",
    "Mexico",
)


def find_player(squads: pd.DataFrame, team: str, player: str) -> pd.Index:
    team_rows = squads[squads["team"] == team]
    key = normalize_name(player)
    exact = team_rows[team_rows["name"].map(normalize_name) == key]
    if len(exact) == 1:
        return exact.index
    contains = team_rows[team_rows["name"].map(normalize_name).str.contains(key, regex=False)]
    return contains.index if len(contains) == 1 else pd.Index([])


def adjust_minutes(squads: pd.DataFrame, player_index, minutes_factor: float) -> pd.DataFrame:
    adjusted = squads.copy()
    row = adjusted.loc[player_index[0]]
    freed = float(row["expected_minutes"]) * (1 - minutes_factor)
    adjusted.loc[player_index, "expected_minutes"] *= minutes_factor
    peers = adjusted[
        (adjusted["team"] == row["team"])
        & (adjusted["position"] == row["position"])
        & (~adjusted.index.isin(player_index))
    ]
    shares = peers["expected_minutes"].to_numpy()
    if shares.sum() <= 0:
        shares = peers["caps"].to_numpy() + 1.0
    adjusted.loc[peers.index, "expected_minutes"] += freed * shares / shares.sum()
    totals = adjusted.groupby("team")["expected_minutes"].transform("sum")
    adjusted["minutes_weight"] = adjusted["expected_minutes"] / totals
    return adjusted


def run_player_scenarios(
    fit,
    n_eff: np.ndarray,
    elo_ratings: dict[str, float],
    baseline: TournamentDetails,
    n_sims: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    from cupcast.run import shrunk_fit  # local import to avoid a cycle

    squads = expected_minutes_per_match(pd.read_csv(SQUADS_CSV))
    baseline_composites = squad_composites(squads)
    if baseline_composites["composite"].isna().all():
        return pd.DataFrame(), [f"{s.name} (needs FBref player data)" for s in SCENARIOS]

    base_table = baseline.table.set_index("team")
    rows, skipped = [], []
    for scenario in SCENARIOS:
        player_index = find_player(squads, scenario.team, scenario.player)
        if player_index.empty:
            skipped.append(f"{scenario.name} (player not found in squad)")
            continue
        adjusted = adjust_minutes(squads, player_index, scenario.minutes_factor)
        composites = squad_composites(adjusted)
        scenario_fit = shrunk_fit(fit, n_eff, elo_ratings, composites)
        details = simulate_tournament(scenario_fit, n_sims=n_sims, seed=seed)
        after = details.table.set_index("team").loc[scenario.team]
        before = base_table.loc[scenario.team]
        rows.append(
            {
                "scenario": scenario.name,
                "team": scenario.team,
                "p_champion_before": before["p_champion"],
                "p_champion_after": after["p_champion"],
                "delta_champion": after["p_champion"] - before["p_champion"],
                "delta_sf": after["p_sf"] - before["p_sf"],
                "delta_final": after["p_final"] - before["p_final"],
            }
        )
    return pd.DataFrame(rows), skipped


def conditional_paths(details: TournamentDetails) -> pd.DataFrame:
    """P(champion | group outcome) per contender, from the same simulations."""
    champion = details.match_winner[104]
    rows = []
    for team in CONDITIONAL_TEAMS:
        team_id = ALL_TEAMS.index(team)
        g = GROUP_LETTERS.index(TEAM_GROUP[team])
        won_group = details.winners[g] == team_id
        runner_up = details.runners[g] == team_id
        third_q = (details.thirds[g] == team_id) & details.qualifies[g]
        is_champion = champion == team_id
        rows.append(
            {
                "team": team,
                "p_champion": float(is_champion.mean()),
                "p_champ_if_group_win": float(is_champion[won_group].mean())
                if won_group.any()
                else np.nan,
                "p_champ_if_runner_up": float(is_champion[runner_up].mean())
                if runner_up.any()
                else np.nan,
                "p_champ_if_third": float(is_champion[third_q].mean())
                if third_q.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
