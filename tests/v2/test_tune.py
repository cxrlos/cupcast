"""Tests for cupcast.v2.tune.optuna_cv."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import math

import numpy as np
import pandas as pd
import pytest

from cupcast.v2.model.prior import dynamic_dc_with_prior
from cupcast.v2.tune.optuna_cv import rolling_origin_folds, tune

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEAMS = ("A", "B", "C", "D", "E", "F")
_RNG = np.random.default_rng(42)


def _make_synthetic_matches() -> pd.DataFrame:
    """120-match synthetic dataset: 6 teams × 5 ordered-pair periods × 4 time slots."""
    rows: list[dict] = []
    dates_by_period = {
        0: pd.Timestamp("2023-01-15"),
        1: pd.Timestamp("2023-04-15"),
        2: pd.Timestamp("2023-07-15"),
        3: pd.Timestamp("2023-10-15"),
    }
    for period, date in dates_by_period.items():
        for home in _TEAMS:
            for away in _TEAMS:
                if home == away:
                    continue
                rows.append(
                    {
                        "date": date,
                        "home": home,
                        "away": away,
                        "home_goals": int(_RNG.poisson(1.3)),
                        "away_goals": int(_RNG.poisson(1.0)),
                        "host_home": False,
                        "period": period,
                    }
                )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _make_zero_clubform(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


# ---------------------------------------------------------------------------
# rolling_origin_folds
# ---------------------------------------------------------------------------


class TestRollingOriginFolds:
    _DATES = pd.Series(
        [
            pd.Timestamp("2023-01-15"),
            pd.Timestamp("2023-04-15"),
            pd.Timestamp("2023-07-15"),
            pd.Timestamp("2023-10-15"),
        ]
    )
    _FOLD_DATES = ["2023-04-01", "2023-10-01"]

    def _folds(self):
        return rolling_origin_folds(self._DATES, self._FOLD_DATES)

    def test_returns_one_fold_per_date(self):
        folds = self._folds()
        assert len(folds) == len(self._FOLD_DATES)

    def test_no_leakage(self):
        folds = self._folds()
        for i, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            fold_ts = pd.Timestamp(self._FOLD_DATES[i])
            max_train_date = self._DATES.iloc[train_idx].max()
            min_test_date = self._DATES.iloc[test_idx].min()
            assert max_train_date < fold_ts, (
                f"Fold {i}: train date {max_train_date} >= fold date {fold_ts}"
            )
            assert min_test_date >= fold_ts, (
                f"Fold {i}: test date {min_test_date} < fold date {fold_ts}"
            )

    def test_non_overlapping(self):
        folds = self._folds()
        for train_idx, test_idx in folds:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_time_ordered(self):
        folds = self._folds()
        for i in range(len(folds) - 1):
            _, test_i = folds[i]
            _, test_next = folds[i + 1]
            if len(test_i) == 0 or len(test_next) == 0:
                continue
            max_i = self._DATES.iloc[test_i].max()
            min_next = self._DATES.iloc[test_next].min()
            assert max_i < min_next, (
                f"Fold {i} test extends past fold {i+1} test start"
            )

    def test_empty_test_idx_when_fold_date_after_all_data(self):
        dates = pd.Series([pd.Timestamp("2023-01-15"), pd.Timestamp("2023-04-15")])
        folds = rolling_origin_folds(dates, ["2023-04-01", "2024-01-01"])
        _, test_idx = folds[1]
        assert len(test_idx) == 0

    def test_train_grows_monotonically(self):
        folds = self._folds()
        prev_len = -1
        for train_idx, _ in folds:
            assert len(train_idx) >= prev_len
            prev_len = len(train_idx)

    def test_timezone_aware_dates(self):
        dates = pd.Series(
            [
                pd.Timestamp("2023-01-15", tz="UTC"),
                pd.Timestamp("2023-04-15", tz="UTC"),
                pd.Timestamp("2023-07-15", tz="UTC"),
            ]
        )
        folds = rolling_origin_folds(dates, ["2023-04-01", "2023-07-01"])
        assert len(folds) == 2
        train_idx_0, _ = folds[0]
        assert len(train_idx_0) == 1  # only 2023-01-15 is before 2023-04-01


# ---------------------------------------------------------------------------
# tune / objective — tiny synthetic problem
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_problem():
    matches = _make_synthetic_matches()
    n = len(set(matches["home"]) | set(matches["away"]))
    cf_att = _make_zero_clubform(n)
    cf_def = _make_zero_clubform(n)
    fold_dates = ["2023-04-01", "2023-10-01"]
    return matches, cf_att, cf_def, fold_dates


class TestTune:
    def test_study_best_value_is_finite(self, tiny_problem):
        matches, cf_att, cf_def, fold_dates = tiny_problem
        study = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=2026,
            _steps_choices=(100,),
        )
        assert math.isfinite(study.best_value), (
            f"best_value={study.best_value} is not finite"
        )

    def test_best_params_has_expected_keys(self, tiny_problem):
        matches, cf_att, cf_def, fold_dates = tiny_problem
        study = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=2026,
            _steps_choices=(100,),
        )
        expected = {"sigma_att_scale", "sigma_def_scale", "tau_prior_scale", "lr", "steps"}
        assert set(study.best_params.keys()) == expected

    def test_reproducible_with_same_seed(self, tiny_problem):
        matches, cf_att, cf_def, fold_dates = tiny_problem
        study_a = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=2026,
            _steps_choices=(100,),
        )
        study_b = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=2026,
            _steps_choices=(100,),
        )
        assert study_a.best_params == study_b.best_params

    def test_different_seeds_may_differ(self, tiny_problem):
        # With n_trials=2 there are only 2 candidates; different seeds may
        # agree by chance, so we just verify both complete without error.
        matches, cf_att, cf_def, fold_dates = tiny_problem
        study_a = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=2026,
            _steps_choices=(100,),
        )
        study_b = tune(
            matches,
            cf_att,
            cf_def,
            fold_dates,
            n_trials=2,
            seed=9999,
            _steps_choices=(100,),
        )
        assert math.isfinite(study_a.best_value)
        assert math.isfinite(study_b.best_value)


# ---------------------------------------------------------------------------
# dynamic_dc_with_prior backward compat
# ---------------------------------------------------------------------------


class TestDynamicDcBackwardCompat:
    """Calling dynamic_dc_with_prior WITHOUT the new args must still work."""

    def test_default_args_unchanged(self):
        import inspect

        sig = inspect.signature(dynamic_dc_with_prior)
        params = sig.parameters
        assert params["sigma_att_scale"].default == pytest.approx(0.3)
        assert params["sigma_def_scale"].default == pytest.approx(0.3)
        assert params["tau_prior_scale"].default == pytest.approx(0.5)

    def test_new_args_are_trailing(self):
        import inspect

        sig = inspect.signature(dynamic_dc_with_prior)
        names = list(sig.parameters.keys())
        assert names.index("sigma_att_scale") > names.index("clubform_defense")
        assert names.index("sigma_def_scale") > names.index("sigma_att_scale")
        assert names.index("tau_prior_scale") > names.index("sigma_def_scale")
