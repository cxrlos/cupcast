import numpy as np
import pandas as pd
import pytest

from cupcast.v1.model.dixon_coles import DixonColesFit
from cupcast.v1.model.shrinkage import (
    apply_shrinkage,
    effective_matches,
    regression_priors,
)


def make_fit(n=20, seed=1):
    rng = np.random.default_rng(seed)
    teams = tuple(f"T{i}" for i in range(n))
    attack = rng.normal(0, 0.4, n)
    return DixonColesFit(
        teams=teams,
        mu=0.1,
        host_advantage=0.25,
        rho=-0.05,
        attack=attack,
        defense=0.8 * attack + rng.normal(0, 0.05, n),
    )


def test_effective_matches_sums_weights_per_team():
    table = pd.DataFrame(
        {"home": ["A", "B", "A"], "away": ["B", "C", "C"]}
    )
    weights = np.array([1.0, 0.5, 0.25])
    n_eff = effective_matches(table, weights, ("A", "B", "C"))
    assert n_eff.tolist() == [1.25, 1.5, 0.75]


def test_priors_recover_linear_elo_relationship():
    fit = make_fit()
    rng = np.random.default_rng(2)
    # Elo constructed to be linear in attack with small noise.
    elo = {
        team: 1500 + 400 * fit.attack[i] + rng.normal(0, 10)
        for i, team in enumerate(fit.teams)
    }
    n_eff = np.full(len(fit.teams), 30.0)
    att_prior, def_prior = regression_priors(fit, n_eff, elo)
    assert np.corrcoef(att_prior, fit.attack)[0, 1] > 0.98
    # defense is attack plus independent noise by construction, so its Elo
    # signal is weaker
    assert np.corrcoef(def_prior, fit.defense)[0, 1] > 0.9


def test_shrinkage_moves_thin_teams_and_leaves_rich_teams():
    fit = make_fit()
    att_prior = np.zeros(len(fit.teams))
    def_prior = np.zeros(len(fit.teams))
    n_eff = np.full(len(fit.teams), 1000.0)
    n_eff[0] = 0.0
    shrunk = apply_shrinkage(fit, n_eff, att_prior, def_prior, k=25.0)
    assert shrunk.attack[0] == pytest.approx(0.0)  # no data -> pure prior
    assert shrunk.attack[5] == pytest.approx(fit.attack[5], rel=0.05)
    assert shrunk.teams == fit.teams
    # blend formula at the midpoint
    mid = apply_shrinkage(fit, np.full(len(fit.teams), 25.0), att_prior, def_prior, k=25.0)
    assert mid.attack[3] == pytest.approx(fit.attack[3] / 2)


def test_composite_enters_priors_when_covered():
    fit = make_fit()
    rng = np.random.default_rng(3)
    elo = {team: 1500 + rng.normal(0, 50) for team in fit.teams}  # uninformative
    composites = pd.DataFrame(
        {
            "team": fit.teams,
            "composite": fit.attack * 2.0,  # composite perfectly tracks attack
            "coverage": 0.9,
        }
    )
    n_eff = np.full(len(fit.teams), 30.0)
    att_prior, _ = regression_priors(fit, n_eff, elo, composites)
    assert np.corrcoef(att_prior, fit.attack)[0, 1] > 0.95
