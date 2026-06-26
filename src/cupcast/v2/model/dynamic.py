"""Quarterly Gaussian-random-walk Dixon-Coles model in NumPyro."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from cupcast.v2.model.dixon_coles import dc_log_tau


def dynamic_dc_model(
    home_idx,
    away_idx,
    period,
    host,
    x,
    y,
    n_teams: int,
    n_periods: int,
    attack_prior_loc=None,
    defense_prior_loc=None,
):
    """Dynamic Dixon-Coles model with a non-centered Gaussian random walk.

    Each team's attack and defence evolve across ``n_periods`` quarters:

        ``attack[i, t] = attack0[i] + σ_α · cumsum(z_att[i, :t+1])``

    Parameters
    ----------
    home_idx, away_idx:
        Integer arrays mapping each match to its home/away team index.
    period:
        Integer array giving the period (0-based quarter index) for each match.
    host:
        Float array; 1.0 when the home side has a genuine venue advantage.
    x, y:
        Observed home/away goal counts.
    n_teams:
        Total number of teams in the index.
    n_periods:
        Total number of quarters covered by the data.
    attack_prior_loc, defense_prior_loc:
        Optional per-team prior means for the initial strength level; ``None`` → 0.
    """
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5))
    rho = numpyro.sample("rho", dist.Uniform(-0.3, 0.3))

    sigma_att = numpyro.sample("sigma_att", dist.HalfNormal(0.3))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(0.3))

    att_loc = (
        jnp.zeros(n_teams) if attack_prior_loc is None else jnp.asarray(attack_prior_loc)
    )
    def_loc = (
        jnp.zeros(n_teams) if defense_prior_loc is None else jnp.asarray(defense_prior_loc)
    )

    with numpyro.plate("teams_init", n_teams):
        attack0 = numpyro.sample("attack0", dist.Normal(att_loc, 1.0))
        defense0 = numpyro.sample("defense0", dist.Normal(def_loc, 1.0))

    # Non-centered innovations: shape (n_teams, n_periods)
    z_att = numpyro.sample(
        "z_att", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2)
    )
    z_def = numpyro.sample(
        "z_def", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2)
    )

    attack = attack0[:, None] + sigma_att * jnp.cumsum(z_att, axis=1)
    defense = defense0[:, None] + sigma_def * jnp.cumsum(z_def, axis=1)

    numpyro.deterministic("attack_traj", attack)
    numpyro.deterministic("attack_now", attack[:, -1])
    numpyro.deterministic("defense_now", defense[:, -1])

    log_lam = mu + attack[home_idx, period] - defense[away_idx, period] + gamma * host
    log_nu = mu + attack[away_idx, period] - defense[home_idx, period]
    lam = jnp.exp(log_lam)
    nu = jnp.exp(log_nu)

    numpyro.sample("x", dist.Poisson(lam), obs=x)
    numpyro.sample("y", dist.Poisson(nu), obs=y)
    numpyro.factor("dc", dc_log_tau(x, y, lam, nu, rho).sum())
