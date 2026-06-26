"""Tests for fit.Posterior + predict functions (Task 4 TDD)."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_posterior():
    from cupcast.v2.model.fit import Posterior

    teams = ("A", "B", "C")
    # A is the strongest attacker; C is the weakest.
    attack = np.array([0.5, 0.0, -0.5])
    defense = np.array([-0.3, 0.0, 0.3])
    return Posterior(teams=teams, attack=attack, defense=defense, mu=0.3, gamma=0.2, rho=-0.05)


def _synthetic_model_args(n_teams=4, n_periods=2, seed=42):
    """Generate synthetic match data from a known parameter set."""
    rng = np.random.default_rng(seed)
    true_attack = np.array([0.7, 0.2, -0.3, -0.1])
    true_defense = np.array([-0.2, 0.0, 0.3, 0.1])
    mu_true = 0.2
    gamma_true = 0.15

    home_idxs, away_idxs, period_list, host_list, xs, ys = [], [], [], [], [], []
    for p in range(n_periods):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam = np.exp(mu_true + true_attack[h] - true_defense[a] + gamma_true)
                nu = np.exp(mu_true + true_attack[a] - true_defense[h])
                home_idxs.append(h)
                away_idxs.append(a)
                period_list.append(p)
                host_list.append(1.0)
                xs.append(int(rng.poisson(lam)))
                ys.append(int(rng.poisson(nu)))

    cf = np.zeros(n_teams)
    return (
        np.array(home_idxs),
        np.array(away_idxs),
        np.array(period_list),
        np.array(host_list, dtype=float),
        np.array(xs),
        np.array(ys),
        n_teams,
        n_periods,
        cf,
        cf,
    )


# ---------------------------------------------------------------------------
# Unit tests on a hand-built Posterior
# ---------------------------------------------------------------------------


def test_posterior_index_built():
    p = _make_small_posterior()
    assert p.index == {"A": 0, "B": 1, "C": 2}


def test_posterior_rate_shape():
    p = _make_small_posterior()
    lam, nu = p.rate("A", "C", host_home=True)
    assert lam > 0 and nu > 0


def test_outcome_probs_sums_to_one():
    from cupcast.v2.model.predict import outcome_probs

    p = _make_small_posterior()
    ph, pd_, pa = outcome_probs(p, "A", "C", host_home=True)
    assert abs(ph + pd_ + pa - 1.0) < 1e-10


def test_score_matrix_shape_and_normalized():
    from cupcast.v2.model.predict import score_matrix

    p = _make_small_posterior()
    m = score_matrix(p, "A", "B", host_home=False)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-10


def test_expected_goals_positive():
    from cupcast.v2.model.predict import expected_goals

    p = _make_small_posterior()
    eg_h, eg_a = expected_goals(p, "A", "C", host_home=True)
    assert eg_h > 0 and eg_a > 0


def test_strong_team_higher_xg():
    """A (strong attack) should generate more xG than C (weak attack)."""
    from cupcast.v2.model.predict import expected_goals

    p = _make_small_posterior()
    eg_h, eg_a = expected_goals(p, "A", "C", host_home=False)
    assert eg_h > eg_a


def test_stronger_team_favoured():
    """outcome_probs should give A > C when A has better attack."""
    from cupcast.v2.model.predict import outcome_probs

    p = _make_small_posterior()
    ph, _pd, pa = outcome_probs(p, "A", "C", host_home=False)
    assert ph > pa


# ---------------------------------------------------------------------------
# End-to-end SVI fit (synthetic data, small, seeded, <15 s)
# ---------------------------------------------------------------------------


def test_fit_svi_returns_posterior():
    from cupcast.v2.model.fit import Posterior, fit_svi
    from cupcast.v2.model.prior import dynamic_dc_with_prior

    teams = ("A", "B", "C", "D")
    model_args = _synthetic_model_args(n_teams=4, n_periods=2)
    post = fit_svi(dynamic_dc_with_prior, model_args, teams, seed=2026, steps=400, lr=0.02)

    assert isinstance(post, Posterior)
    assert post.teams == teams
    assert len(post.attack) == 4
    assert len(post.defense) == 4
    assert np.all(np.isfinite(post.attack))
    assert np.all(np.isfinite(post.defense))
    assert np.isfinite(post.mu)
    assert np.isfinite(post.gamma)
    assert np.isfinite(post.rho)


def test_fit_svi_strong_team_wins():
    """After SVI fit on data generated with A as strongest, A should beat C."""
    from cupcast.v2.model.fit import fit_svi
    from cupcast.v2.model.predict import outcome_probs
    from cupcast.v2.model.prior import dynamic_dc_with_prior

    teams = ("A", "B", "C", "D")
    model_args = _synthetic_model_args(n_teams=4, n_periods=2)
    post = fit_svi(dynamic_dc_with_prior, model_args, teams, seed=2026, steps=400, lr=0.02)

    ph, _pd, pa = outcome_probs(post, "A", "C", host_home=True)
    assert ph > pa, f"A should beat C but p_home={ph:.3f}, p_away={pa:.3f}"


# ---------------------------------------------------------------------------
# host_away extension (Task 1)
# ---------------------------------------------------------------------------


def test_rate_host_away_raises_nu():
    """host_away=True must give a higher away rate (nu) than host_away=False."""
    p = _make_small_posterior()
    _, nu_base = p.rate("A", "B", host_home=False, host_away=False)
    _, nu_boost = p.rate("A", "B", host_home=False, host_away=True)
    assert nu_boost > nu_base, f"Expected nu_boost > nu_base but {nu_boost} <= {nu_base}"


def test_rate_host_away_does_not_affect_lam():
    """host_away must not change the home rate (lam)."""
    p = _make_small_posterior()
    lam_base, _ = p.rate("A", "B", host_home=False, host_away=False)
    lam_boost, _ = p.rate("A", "B", host_home=False, host_away=True)
    assert abs(lam_boost - lam_base) < 1e-12


def test_score_matrix_host_away_shifts_mass():
    """score_matrix(..., host_away=True) should increase p_away relative to host_away=False."""
    from cupcast.v2.model.predict import outcome_probs

    p = _make_small_posterior()
    _, _, pa_base = outcome_probs(p, "A", "B", host_home=False, host_away=False)
    _, _, pa_boost = outcome_probs(p, "A", "B", host_home=False, host_away=True)
    assert pa_boost > pa_base, f"Expected p_away to rise but {pa_boost} <= {pa_base}"


def test_score_matrix_host_away_default_false():
    """host_away defaults to False — must match explicit host_away=False."""
    from cupcast.v2.model.predict import score_matrix

    p = _make_small_posterior()
    m_default = score_matrix(p, "A", "B", host_home=False)
    m_explicit = score_matrix(p, "A", "B", host_home=False, host_away=False)
    assert np.allclose(m_default, m_explicit)


def test_expected_goals_host_away_raises_away_xg():
    """expected_goals with host_away=True must produce higher away xG."""
    from cupcast.v2.model.predict import expected_goals

    p = _make_small_posterior()
    _, eg_a_base = expected_goals(p, "A", "B", host_home=False, host_away=False)
    _, eg_a_boost = expected_goals(p, "A", "B", host_home=False, host_away=True)
    assert eg_a_boost > eg_a_base
