"""Dixon-Coles likelihood and static NumPyro model."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd


def dc_log_tau(x, y, lam, nu, rho):
    """Log of the Dixon-Coles low-score dependence correction τ(x,y;λ,ν,ρ).

    Vectorized over JAX arrays via ``jnp.where`` so the function is jit-safe.
    Returns ``log(τ)`` clipped at ``1e-10`` to avoid ``-inf``.
    """
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau = jnp.where(
        m00,
        1.0 - lam * nu * rho,
        jnp.where(
            m01,
            1.0 + lam * rho,
            jnp.where(
                m10,
                1.0 + nu * rho,
                jnp.where(m11, 1.0 - rho, jnp.ones_like(lam)),
            ),
        ),
    )
    return jnp.log(jnp.clip(tau, 1e-10, None))


def team_index(matches: pd.DataFrame) -> tuple[tuple[str, ...], list[int], list[int]]:
    """Map team names in ``matches`` to a contiguous integer index.

    Returns
    -------
    teams:
        Sorted tuple of unique team names.
    home_idx, away_idx:
        Integer index arrays parallel to ``matches``.
    """
    teams = tuple(sorted(set(matches["home"]) | set(matches["away"])))
    index = {t: i for i, t in enumerate(teams)}
    home_idx = matches["home"].map(index).tolist()
    away_idx = matches["away"].map(index).tolist()
    return teams, home_idx, away_idx


def static_dc_model(home_idx, away_idx, host, x, y, n_teams):
    """Static Dixon-Coles model in NumPyro.

    Priors: global ``mu``, home-advantage ``gamma``, DC correlation ``rho``;
    per-team ``attack`` and ``defense`` drawn from hierarchical half-normals.
    Observed goals are Poisson-distributed and augmented with the DC factor.
    """
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5))
    rho = numpyro.sample("rho", dist.Uniform(-0.3, 0.3))

    sd_att = numpyro.sample("sd_att", dist.HalfNormal(1.0))
    sd_def = numpyro.sample("sd_def", dist.HalfNormal(1.0))

    with numpyro.plate("teams", n_teams):
        attack = numpyro.sample("attack", dist.Normal(0.0, sd_att))
        defense = numpyro.sample("defense", dist.Normal(0.0, sd_def))

    log_lam = mu + attack[home_idx] - defense[away_idx] + gamma * host
    log_nu = mu + attack[away_idx] - defense[home_idx]
    lam = jnp.exp(log_lam)
    nu = jnp.exp(log_nu)

    numpyro.sample("x", dist.Poisson(lam), obs=x)
    numpyro.sample("y", dist.Poisson(nu), obs=y)
    numpyro.factor("dc", dc_log_tau(x, y, lam, nu, rho).sum())
