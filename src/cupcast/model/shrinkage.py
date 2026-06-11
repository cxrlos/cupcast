from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from cupcast.model.dixon_coles import DixonColesFit

# Effective-match count at which fitted ratings and the prior carry equal
# weight. Tuned on rolling out-of-sample folds 2022-2026 (n=4167): k=10 gave
# the best log-loss (0.8615 vs 0.8623 unshrunk); k>=25 was worse than no
# shrinkage at all. Light shrinkage only — the prior earns its keep on
# thin-history teams and must not drag established ones.
DEFAULT_K = 10.0


def effective_matches(table: pd.DataFrame, weights: np.ndarray, teams: tuple[str, ...]):
    counts = dict.fromkeys(teams, 0.0)
    for side in ("home", "away"):
        sums = pd.Series(weights, index=table.index).groupby(table[side]).sum()
        for team, value in sums.items():
            if team in counts:
                counts[team] += float(value)
    return np.array([counts[t] for t in teams])


def regression_priors(
    fit: DixonColesFit,
    n_eff: np.ndarray,
    elo: dict[str, float],
    composites: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-team empirical priors for attack and defense.

    Base tier regresses fitted ratings on Elo (weighted by effective sample, so
    well-identified teams define the relationship). Where a squad composite
    with adequate coverage exists, it enters as a second regressor. Returns
    predicted (attack_prior, defense_prior) for every team in the fit.
    """
    elo_values = np.array([elo.get(team, np.nan) for team in fit.teams])
    elo_z = (elo_values - np.nanmean(elo_values)) / np.nanstd(elo_values)
    elo_z = np.nan_to_num(elo_z)

    composite = np.full(len(fit.teams), np.nan)
    if composites is not None and not composites.empty:
        lookup = composites.set_index("team")
        for i, team in enumerate(fit.teams):
            if team in lookup.index:
                row = lookup.loc[team]
                if pd.notna(row["composite"]) and row["coverage"] >= 0.35:
                    composite[i] = row["composite"]

    weights = np.sqrt(n_eff)

    def weighted_fit(target: np.ndarray) -> np.ndarray:
        prediction = np.empty(len(target))
        covered = ~np.isnan(composite)
        base_design = np.column_stack([np.ones(len(target)), elo_z])
        coef, *_ = np.linalg.lstsq(
            base_design * weights[:, None], target * weights, rcond=None
        )
        prediction[:] = base_design @ coef
        if covered.sum() >= 12:
            design = np.column_stack(
                [np.ones(covered.sum()), elo_z[covered], composite[covered]]
            )
            coef_full, *_ = np.linalg.lstsq(
                design * weights[covered, None], target[covered] * weights[covered], rcond=None
            )
            prediction[covered] = design @ coef_full
        return prediction

    return weighted_fit(fit.attack), weighted_fit(fit.defense)


def apply_shrinkage(
    fit: DixonColesFit,
    n_eff: np.ndarray,
    attack_prior: np.ndarray,
    defense_prior: np.ndarray,
    k: float = DEFAULT_K,
) -> DixonColesFit:
    blend = n_eff / (n_eff + k)
    return replace(
        fit,
        attack=blend * fit.attack + (1 - blend) * attack_prior,
        defense=blend * fit.defense + (1 - blend) * defense_prior,
    )
