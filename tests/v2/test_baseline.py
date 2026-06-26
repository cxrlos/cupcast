"""Tests for penaltyblog baseline wrapper (Task 4 TDD)."""

import numpy as np
import pandas as pd


def _synthetic_matches(n: int = 80, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    records = []
    while len(records) < n:
        h, a = rng.choice(len(teams), size=2, replace=False)
        records.append(
            {
                "home": teams[h],
                "away": teams[a],
                "home_goals": int(rng.poisson(1.5)),
                "away_goals": int(rng.poisson(1.1)),
            }
        )
    return pd.DataFrame(records)


def test_outcome_probs_pb_is_simplex():
    from cupcast.v2.model.baseline import fit_penaltyblog_dc, outcome_probs_pb

    matches = _synthetic_matches(80)
    fit = fit_penaltyblog_dc(matches)
    ph, pd_, pa = outcome_probs_pb(fit, "TeamA", "TeamB")

    assert abs(ph + pd_ + pa - 1.0) < 1e-6, f"probs sum to {ph+pd_+pa}"
    assert ph > 0 and pd_ > 0 and pa > 0


def test_outcome_probs_pb_nonnegative():
    from cupcast.v2.model.baseline import fit_penaltyblog_dc, outcome_probs_pb

    matches = _synthetic_matches(80)
    fit = fit_penaltyblog_dc(matches)
    for home, away in [("TeamA", "TeamC"), ("TeamB", "TeamD"), ("TeamC", "TeamA")]:
        ph, pd_, pa = outcome_probs_pb(fit, home, away)
        assert ph >= 0 and pd_ >= 0 and pa >= 0
