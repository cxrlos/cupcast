import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from cupcast.model.dixon_coles import DixonColesFit, build_objective, fit_dixon_coles


def synthetic_matches(n_teams=12, n_matches=4000, mu=0.1, gamma=0.3, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defense = rng.normal(0, 0.3, n_teams)
    defense -= defense.mean()
    h = rng.integers(0, n_teams, n_matches)
    a = (h + rng.integers(1, n_teams, n_matches)) % n_teams
    host = rng.random(n_matches) < 0.3
    lam = np.exp(mu + attack[h] - defense[a] + gamma * host)
    nu = np.exp(mu + attack[a] - defense[h])
    frame = pd.DataFrame(
        {
            "home": [teams[i] for i in h],
            "away": [teams[i] for i in a],
            "home_goals": rng.poisson(lam),
            "away_goals": rng.poisson(nu),
            "host_home": host,
        }
    )
    return frame, teams, attack, defense


def test_gradient_matches_finite_differences():
    matches, *_ = synthetic_matches(n_teams=5, n_matches=120, seed=3)
    objective, theta0, _, _ = build_objective(matches)
    rng = np.random.default_rng(11)
    theta = theta0 + rng.normal(0, 0.05, theta0.size)
    _, analytic = objective(theta)
    numeric = approx_fprime(theta, lambda t: objective(t)[0], 1e-7)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-3)


def test_fit_recovers_synthetic_parameters():
    matches, teams, attack, defense = synthetic_matches()
    fit = fit_dixon_coles(matches)
    order = [fit.teams.index(t) for t in teams]
    assert np.corrcoef(attack, fit.attack[order])[0, 1] > 0.9
    assert np.corrcoef(defense, fit.defense[order])[0, 1] > 0.9
    assert fit.mu == pytest.approx(0.1, abs=0.08)
    assert fit.host_advantage == pytest.approx(0.3, abs=0.1)
    assert abs(fit.rho) < 0.06  # data generated with independent Poissons


def make_fit(rho=0.0, mu=0.2):
    return DixonColesFit(
        teams=("A", "B"),
        mu=mu,
        host_advantage=0.3,
        rho=rho,
        attack=np.zeros(2),
        defense=np.zeros(2),
    )


def test_score_matrix_is_a_distribution_and_symmetric_for_equal_teams():
    fit = make_fit()
    matrix = fit.score_matrix("A", "B")
    assert matrix.sum() == pytest.approx(1.0)
    home, draw, away = fit.outcome_probs("A", "B")
    assert home + draw + away == pytest.approx(1.0)
    assert home == pytest.approx(away)


def test_negative_rho_inflates_low_draws():
    base = make_fit(rho=0.0).score_matrix("A", "B")
    adjusted = make_fit(rho=-0.1).score_matrix("A", "B")
    assert adjusted[0, 0] > base[0, 0]
    assert adjusted[1, 1] > base[1, 1]
    assert adjusted[0, 1] < base[0, 1]


def test_host_advantage_raises_home_rate():
    fit = make_fit()
    neutral_lam, neutral_nu = fit.rates("A", "B", host=False)
    host_lam, host_nu = fit.rates("A", "B", host=True)
    assert host_lam == pytest.approx(neutral_lam * np.exp(0.3))
    assert host_nu == neutral_nu


def test_unknown_team_raises():
    fit = make_fit()
    with pytest.raises(KeyError, match="not in fit"):
        fit.rates("A", "Narnia")
