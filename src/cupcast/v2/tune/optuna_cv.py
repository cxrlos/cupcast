"""Optuna rolling-origin hyperparameter tuner for the dynamic DC model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd

from cupcast.v2.model.dixon_coles import team_index
from cupcast.v2.model.fit import fit_svi
from cupcast.v2.model.predict import outcome_probs
from cupcast.v2.model.prior import dynamic_dc_with_prior
from cupcast.v2.model.validate import log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rolling_origin_folds(
    dates: pd.Series, fold_dates: list[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return rolling-origin (train, test) index arrays, one per fold date.

    Train indices are strictly before each fold date; test indices cover
    ``[fold_date, next_fold_date)``.  The last fold's test set extends to
    the end of the series.

    Parameters
    ----------
    dates:
        Series of match dates (pd.Timestamp, timezone-aware or naive).
    fold_dates:
        ISO date strings defining fold boundaries.

    Returns
    -------
    List of ``(train_idx, test_idx)`` numpy integer-index arrays.
    """
    if dates.empty:
        return []

    tz = getattr(dates.iloc[0], "tzinfo", None)
    if tz is not None:
        ts_folds = [pd.Timestamp(d, tz="UTC") for d in fold_dates]
    else:
        ts_folds = [pd.Timestamp(d) for d in fold_dates]

    all_idx = np.arange(len(dates))
    result: list[tuple[np.ndarray, np.ndarray]] = []

    for i, cutoff in enumerate(ts_folds):
        next_cutoff = ts_folds[i + 1] if i + 1 < len(ts_folds) else None
        train_idx = all_idx[dates < cutoff]
        if next_cutoff is not None:
            test_idx = all_idx[(dates >= cutoff) & (dates < next_cutoff)]
        else:
            test_idx = all_idx[dates >= cutoff]
        result.append((train_idx, test_idx))

    return result


def objective(
    trial: optuna.Trial,
    matches: pd.DataFrame,
    clubform_attack: np.ndarray,
    clubform_defense: np.ndarray,
    fold_dates: list[str],
    base_seed: int = 2026,
    _steps_choices: tuple[int, ...] = (1500, 2000, 2500),
) -> float:
    """Optuna objective: mean OOS log-loss over rolling-origin folds.

    Samples ``sigma_att_scale``, ``sigma_def_scale``, ``tau_prior_scale``,
    ``lr``, and ``steps`` for each trial, fits the dynamic DC model per fold
    via SVI, and returns the mean log-loss on held-out test matches.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    matches:
        DataFrame with columns ``date, home, away, home_goals, away_goals,
        host_home, period``.
    clubform_attack, clubform_defense:
        Standardized clubform composites aligned to ``team_index(matches)``.
    fold_dates:
        ISO date strings for fold boundaries.
    base_seed:
        Base RNG seed; each fold uses ``base_seed + fold_index``.
    _steps_choices:
        Categorical choices for SVI step count (override in tests for speed).
    """
    sigma_att_scale = trial.suggest_float("sigma_att_scale", 0.05, 0.6, log=True)
    sigma_def_scale = trial.suggest_float("sigma_def_scale", 0.05, 0.6, log=True)
    tau_prior_scale = trial.suggest_float("tau_prior_scale", 0.3, 1.5)
    lr = trial.suggest_float("lr", 0.005, 0.05, log=True)
    steps = trial.suggest_categorical("steps", list(_steps_choices))

    global_teams, _, _ = team_index(matches)
    global_to_idx = {t: i for i, t in enumerate(global_teams)}

    folds = rolling_origin_folds(matches["date"], fold_dates)

    fold_losses: list[float] = []
    for fold_num, (train_idx, test_idx) in enumerate(folds):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train = matches.iloc[train_idx]
        test = matches.iloc[test_idx]

        fold_teams, fold_home_idx, fold_away_idx = team_index(train)
        fold_set = set(fold_teams)

        fold_cf_att = np.array([clubform_attack[global_to_idx[t]] for t in fold_teams])
        fold_cf_def = np.array([clubform_defense[global_to_idx[t]] for t in fold_teams])

        n_periods = int(train["period"].max()) + 1

        model_args = (
            jnp.array(fold_home_idx),
            jnp.array(fold_away_idx),
            jnp.array(train["period"].to_numpy(dtype=int)),
            jnp.array(train["host_home"].to_numpy(dtype=float)),
            jnp.array(train["home_goals"].to_numpy(dtype=int)),
            jnp.array(train["away_goals"].to_numpy(dtype=int)),
            len(fold_teams),
            n_periods,
            jnp.array(fold_cf_att),
            jnp.array(fold_cf_def),
            float(sigma_att_scale),
            float(sigma_def_scale),
            float(tau_prior_scale),
        )

        posterior = fit_svi(
            dynamic_dc_with_prior,
            model_args,
            fold_teams,
            seed=base_seed + fold_num,
            steps=int(steps),
            lr=float(lr),
        )

        P_rows: list[list[float]] = []
        y_rows: list[int] = []
        for _, row in test.iterrows():
            home, away = str(row["home"]), str(row["away"])
            if home not in fold_set or away not in fold_set:
                continue
            host = bool(row.get("host_home", False))
            probs = outcome_probs(posterior, home, away, host_home=host)
            P_rows.append(list(probs))
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            y_rows.append(0 if hg > ag else (1 if hg == ag else 2))

        if not P_rows:
            continue

        fold_losses.append(log_loss(np.array(P_rows), np.array(y_rows)))

    return float(np.mean(fold_losses)) if fold_losses else float("inf")


def tune(
    matches: pd.DataFrame,
    clubform_attack: np.ndarray,
    clubform_defense: np.ndarray,
    fold_dates: list[str],
    n_trials: int = 40,
    seed: int = 2026,
    storage=None,
    _steps_choices: tuple[int, ...] = (1500, 2000, 2500),
) -> optuna.Study:
    """Run an Optuna hyperparameter search and return the completed study.

    Uses ``TPESampler(seed=seed)`` for reproducibility.  When ``storage`` is
    provided the study is checkpointed and resumes automatically if the same
    ``study_name`` already exists.

    Parameters
    ----------
    matches:
        Pre-cutoff internationals (WC2026 must NOT be present).
    clubform_attack, clubform_defense:
        Aligned to ``team_index(matches)`` — see :func:`objective`.
    fold_dates:
        ISO date strings for fold boundaries.
    n_trials:
        Number of Optuna trials.
    seed:
        Random seed for the TPE sampler and model fits.
    storage:
        Optional Optuna storage URL for checkpoint/resume.
    _steps_choices:
        Passed through to :func:`objective` (override in tests for speed).
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    create_kwargs: dict = {"direction": "minimize", "sampler": sampler}
    if storage is not None:
        create_kwargs["storage"] = storage
        create_kwargs["study_name"] = f"cupcast_tune_{seed}"
        create_kwargs["load_if_exists"] = True

    study = optuna.create_study(**create_kwargs)
    study.optimize(
        lambda t: objective(
            t, matches, clubform_attack, clubform_defense, fold_dates, seed, _steps_choices
        ),
        n_trials=n_trials,
    )
    return study
