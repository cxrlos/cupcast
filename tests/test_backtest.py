import numpy as np
import pandas as pd

from cupcast.v1.validate.backtest import fit_as_of, rolling_folds


def test_rolling_folds_are_contiguous_and_ordered():
    folds = rolling_folds("2022-01-01", "2024-01-01", step_months=6)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_until < fold.eval_until
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.eval_until == later.train_until


def test_fit_as_of_never_sees_future_matches():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=200, freq="3D")
    teams = [f"T{i}" for i in range(8)]
    home = rng.choice(teams, len(dates))
    away = np.array([rng.choice([t for t in teams if t != h]) for h in home])
    table = pd.DataFrame(
        {
            "date": dates,
            "home": home,
            "away": away,
            "home_goals": rng.poisson(1.2, len(dates)),
            "away_goals": rng.poisson(1.0, len(dates)),
            "neutral": False,
            "friendly": False,
        }
    )
    cutoff = pd.Timestamp("2023-06-01")
    fit_full = fit_as_of(table, cutoff, half_life_years=3.0, friendly_weight=0.5)
    fit_trimmed = fit_as_of(
        table[table["date"] < cutoff], cutoff, half_life_years=3.0, friendly_weight=0.5
    )
    assert fit_full.teams == fit_trimmed.teams
    assert np.allclose(fit_full.attack, fit_trimmed.attack, atol=1e-6)
    assert fit_full.mu == fit_trimmed.mu
