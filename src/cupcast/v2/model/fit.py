"""SVI and NUTS fitters returning a Posterior for the dynamic DC model."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam


@dataclass
class Posterior:
    """Posterior-mean parameter bundle from a fitted dynamic DC model.

    ``attack`` and ``defense`` are current-period strengths
    (``attack_now``/``defense_now`` from the model deterministics).
    """

    teams: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    mu: float
    gamma: float
    rho: float
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.teams)}

    def rate(
        self, home: str, away: str, host_home: bool, host_away: bool = False
    ) -> tuple[float, float]:
        """Expected goals per team using posterior-mean parameters.

        ``lam = exp(mu + attack[h] - defense[a] + gamma * host_home)``
        ``nu  = exp(mu + attack[a] - defense[h] + gamma * host_away)``
        """
        h, a = self.index[home], self.index[away]
        lam = float(np.exp(self.mu + self.attack[h] - self.defense[a] + self.gamma * host_home))
        nu = float(np.exp(self.mu + self.attack[a] - self.defense[h] + self.gamma * host_away))
        return lam, nu


def _extract_posterior(samples: dict, teams: tuple[str, ...]) -> Posterior:
    attack = np.array(jnp.mean(samples["attack_now"], axis=0))
    defense = np.array(jnp.mean(samples["defense_now"], axis=0))
    mu = float(jnp.mean(samples["mu"]))
    gamma = float(jnp.mean(samples["gamma"]))
    rho = float(jnp.mean(samples["rho"]))
    return Posterior(teams=teams, attack=attack, defense=defense, mu=mu, gamma=gamma, rho=rho)


def fit_svi(
    model,
    model_args: tuple,
    teams: tuple[str, ...],
    seed: int = 2026,
    steps: int = 3000,
    lr: float = 0.02,
) -> Posterior:
    """Fit ``model`` via SVI (AutoNormal guide, Adam) and return a Posterior.

    Draws 200 posterior samples via ``Predictive`` and takes means of
    ``attack_now``, ``defense_now``, ``mu``, ``gamma``, ``rho``.
    """
    # init_to_median starts the guide at the prior median (strengths ~0, rates ~1),
    # avoiding the exp() overflow a random init can trigger on this model.
    guide = AutoNormal(model, init_loc_fn=init_to_median, init_scale=0.1)
    svi = SVI(model, guide, Adam(lr), loss=Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(seed), steps, *model_args)
    params = svi_result.params

    _sites = ["attack_now", "defense_now", "mu", "gamma", "rho"]
    predictive = Predictive(model, guide=guide, params=params, num_samples=200, return_sites=_sites)
    samples = predictive(jax.random.PRNGKey(seed + 1), *model_args)
    return _extract_posterior(samples, teams)


def fit_nuts(
    model,
    model_args: tuple,
    teams: tuple[str, ...],
    seed: int = 2026,
    num_warmup: int = 1000,
    num_samples: int = 1000,
) -> Posterior:
    """Fit ``model`` via NUTS and return a Posterior from posterior means."""
    kernel = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *model_args)
    samples = mcmc.get_samples()
    return _extract_posterior(samples, teams)
