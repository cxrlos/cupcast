"""Clubform-informed learned prior (f_θ) for the dynamic Dixon-Coles model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from cupcast.v2.model.dixon_coles import dc_log_tau


def clubform_prior_locs(
    clubform: pd.DataFrame, teams: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Align and standardize clubform composites to the model's team index.

    Teams absent from ``clubform`` receive 0.0 (the standardized mean).

    Parameters
    ----------
    clubform:
        DataFrame with at least ``team``, ``attack``, and ``defense`` columns.
    teams:
        Ordered tuple of team names matching the model's integer index.

    Returns
    -------
    attack_loc, defense_loc:
        Float arrays of length ``len(teams)`` standardized to mean 0 / sd 1
        over teams present in ``clubform``; missing teams map to 0.0.
    """
    cf = clubform.set_index("team") if "team" in clubform.columns else clubform
    present = [t for t in teams if t in cf.index]

    if not present:
        n = len(teams)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    att_vals = cf.loc[present, "attack"].values.astype(float)
    def_vals = cf.loc[present, "defense"].values.astype(float)

    att_mean = float(att_vals.mean())
    att_std = float(att_vals.std(ddof=1)) if len(att_vals) > 1 else 1.0
    def_mean = float(def_vals.mean())
    def_std = float(def_vals.std(ddof=1)) if len(def_vals) > 1 else 1.0

    att_std = max(att_std, 1e-8)
    def_std = max(def_std, 1e-8)

    att_norm = {t: float((cf.loc[t, "attack"] - att_mean) / att_std) for t in present}
    def_norm = {t: float((cf.loc[t, "defense"] - def_mean) / def_std) for t in present}

    attack_loc = np.array([att_norm.get(t, 0.0) for t in teams], dtype=float)
    defense_loc = np.array([def_norm.get(t, 0.0) for t in teams], dtype=float)

    return attack_loc, defense_loc


def dynamic_dc_with_prior(
    home_idx,
    away_idx,
    period,
    host,
    x,
    y,
    n_teams: int,
    n_periods: int,
    clubform_attack,
    clubform_defense,
    sigma_att_scale: float = 0.3,
    sigma_def_scale: float = 0.3,
    tau_prior_scale: float = 0.5,
):
    """Dynamic Dixon-Coles model with a learned linear clubform prior (f_θ).

    Extends the quarterly-GRW model so the initial team strength is drawn from
    a learned linear function of the per-team clubform composites::

        attack0[i]  ~ N(a0 + a1 · clubform_attack[i],  tau_prior)
        defense0[i] ~ N(d0 + d1 · clubform_defense[i], tau_prior)

    Data-poor teams remain close to this prior; data-rich teams' random walks
    move them away.  The learned slopes ``a1`` and ``d1`` are inspectable as
    sample sites in the posterior.

    Parameters
    ----------
    home_idx, away_idx:
        Integer arrays mapping each match to its home/away team index.
    period:
        Integer array giving the 0-based quarter index for each match.
    host:
        Float array; 1.0 when the home side has a genuine venue advantage.
    x, y:
        Observed home/away goal counts.
    n_teams:
        Total number of teams in the index.
    n_periods:
        Total number of quarters covered by the data.
    clubform_attack, clubform_defense:
        Standardized clubform composites aligned to the team index, length
        ``n_teams``.  Produce these with :func:`clubform_prior_locs`.
    sigma_att_scale:
        Scale of the ``HalfNormal`` prior on the attack GRW innovation SD.
    sigma_def_scale:
        Scale of the ``HalfNormal`` prior on the defense GRW innovation SD.
    tau_prior_scale:
        Scale of the ``HalfNormal`` prior on the clubform-prior dispersion.
    """
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5))
    rho = numpyro.sample("rho", dist.Uniform(-0.3, 0.3))

    sigma_att = numpyro.sample("sigma_att", dist.HalfNormal(sigma_att_scale))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(sigma_def_scale))

    # f_θ linear map coefficients.
    a0 = numpyro.sample("a0", dist.Normal(0.0, 1.0))
    a1 = numpyro.sample("a1", dist.Normal(0.0, 1.0))
    d0 = numpyro.sample("d0", dist.Normal(0.0, 1.0))
    d1 = numpyro.sample("d1", dist.Normal(0.0, 1.0))
    tau_prior = numpyro.sample("tau_prior", dist.HalfNormal(tau_prior_scale))

    cf_att = jnp.asarray(clubform_attack)
    cf_def = jnp.asarray(clubform_defense)

    with numpyro.plate("teams_init", n_teams):
        attack0 = numpyro.sample("attack0", dist.Normal(a0 + a1 * cf_att, tau_prior))
        defense0 = numpyro.sample("defense0", dist.Normal(d0 + d1 * cf_def, tau_prior))

    # Non-centered quarterly GRW: shape (n_teams, n_periods).
    z_att = numpyro.sample(
        "z_att", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2)
    )
    z_def = numpyro.sample(
        "z_def", dist.Normal(0.0, 1.0).expand([n_teams, n_periods]).to_event(2)
    )

    attack = attack0[:, None] + sigma_att * jnp.cumsum(z_att, axis=1)
    defense = defense0[:, None] + sigma_def * jnp.cumsum(z_def, axis=1)

    numpyro.deterministic("attack_now", attack[:, -1])
    numpyro.deterministic("defense_now", defense[:, -1])

    log_lam = mu + attack[home_idx, period] - defense[away_idx, period] + gamma * host
    log_nu = mu + attack[away_idx, period] - defense[home_idx, period]
    lam = jnp.exp(log_lam)
    nu = jnp.exp(log_nu)

    numpyro.sample("x", dist.Poisson(lam), obs=x)
    numpyro.sample("y", dist.Poisson(nu), obs=y)
    numpyro.factor("dc", dc_log_tau(x, y, lam, nu, rho).sum())
