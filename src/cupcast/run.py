from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from cupcast.features.matches import build_match_table
from cupcast.features.squad_strength import squad_composites
from cupcast.model.dixon_coles import fit_dixon_coles
from cupcast.model.shrinkage import (
    apply_shrinkage,
    effective_matches,
    regression_priors,
)
from cupcast.model.weights import match_weights
from cupcast.ratings.history import elo_history
from cupcast.sim.monte_carlo import run_tournament

AS_OF = pd.Timestamp("2026-06-11")
OUTPUTS = Path("outputs")


def build_fit(as_of: pd.Timestamp = AS_OF):
    table = build_match_table(since="2015-01-01")
    table = table[table["date"] < as_of].copy()
    table["host_home"] = ~table["neutral"].astype(bool)
    weights = match_weights(table["date"], as_of, table["friendly"])
    fit = fit_dixon_coles(table, weights)

    elo_table = build_match_table()  # full history for converged ratings
    elo_ratings, _ = elo_history(elo_table[elo_table["date"] < as_of])

    try:
        composites = squad_composites()
        covered = int((composites["coverage"] >= 0.35).sum())
        print(f"squad composites: {covered}/48 teams above coverage threshold")
    except FileNotFoundError:
        composites = None
        print("squad composites unavailable (no squads.csv); Elo-only priors")
    if composites is not None and composites["composite"].isna().all():
        print("no FBref player data found; falling back to Elo-only priors")
        composites = None

    n_eff = effective_matches(table, weights, fit.teams)
    attack_prior, defense_prior = regression_priors(fit, n_eff, elo_ratings, composites)
    return apply_shrinkage(fit, n_eff, attack_prior, defense_prior)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(prog="cupcast-run", description="Full forecast pipeline")
    parser.add_argument("--sims", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    fit = build_fit()
    result = run_tournament(fit, n_sims=args.sims, seed=args.seed)

    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / "simulation_results.csv"
    result.round(5).to_csv(out, index=False)
    print(f"\n{args.sims} simulations -> {out}")
    cols = ["team", "p_qualify", "p_sf", "p_final", "p_champion", "p_runner_up", "p_podium_third"]
    print(result[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    champion = result.nlargest(1, "p_champion")["team"].iloc[0]
    runner = (
        result[result["team"] != champion].nlargest(1, "p_runner_up")["team"].iloc[0]
    )
    third = (
        result[~result["team"].isin([champion, runner])]
        .nlargest(1, "p_podium_third")["team"]
        .iloc[0]
    )
    print(f"\nmodal podium: 1. {champion}  2. {runner}  3. {third}")
    print(f"reproduce: uv run python -m cupcast.run --sims {args.sims} --seed {args.seed}")


if __name__ == "__main__":
    main()
