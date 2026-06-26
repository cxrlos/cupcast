"""TDD for validate.py scorers and rolling_score (Plan 4 Task 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_P(y: np.ndarray, K: int = 3) -> np.ndarray:
    """One-hot probability array giving perfect predictions."""
    P = np.zeros((len(y), K))
    P[np.arange(len(y)), y] = 1.0
    return P


def _uniform_P(n: int, K: int = 3) -> np.ndarray:
    return np.full((n, K), 1.0 / K)


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------


class TestLogLoss:
    def test_perfect_prediction_zero(self):
        from cupcast.v2.model.validate import log_loss

        y = np.array([0, 1, 2, 0, 1])
        assert log_loss(_perfect_P(y), y) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_known_value(self):
        from cupcast.v2.model.validate import log_loss

        y = np.array([0, 1, 2])
        # -log(1/3) = log(3)
        assert log_loss(_uniform_P(3), y) == pytest.approx(np.log(3), rel=1e-6)

    def test_clips_zeros_no_inf(self):
        from cupcast.v2.model.validate import log_loss

        # Probability 0 assigned to the realized outcome; must not return -inf.
        P = np.array([[0.0, 0.5, 0.5]])
        y = np.array([0])
        result = log_loss(P, y)
        assert np.isfinite(result)
        assert result > 0

    def test_single_match(self):
        from cupcast.v2.model.validate import log_loss

        P = np.array([[0.6, 0.3, 0.1]])
        y = np.array([0])
        assert log_loss(P, y) == pytest.approx(-np.log(0.6), rel=1e-6)


# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------


class TestBrier:
    def test_perfect_prediction_zero(self):
        from cupcast.v2.model.validate import brier

        y = np.array([0, 1, 2])
        assert brier(_perfect_P(y), y) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_known_value(self):
        from cupcast.v2.model.validate import brier

        # For any y, P=(1/3,1/3,1/3): (1/3-1)^2 + 2*(1/3-0)^2 = 4/9 + 2/9 = 2/3
        y = np.array([0, 1, 2, 0])
        assert brier(_uniform_P(4), y) == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_single_match_manual(self):
        from cupcast.v2.model.validate import brier

        P = np.array([[0.7, 0.2, 0.1]])
        y = np.array([0])
        # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14
        assert brier(P, y) == pytest.approx(0.14, rel=1e-6)


# ---------------------------------------------------------------------------
# rps
# ---------------------------------------------------------------------------


class TestRPS:
    def test_perfect_prediction_zero(self):
        from cupcast.v2.model.validate import rps

        y = np.array([0, 1, 2])
        assert rps(_perfect_P(y), y) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_home_win(self):
        from cupcast.v2.model.validate import rps

        # cumP=[1/3,2/3,1], cumO=[1,1,1]
        # sum = (2/3)^2 + (1/3)^2 + 0 = 5/9; /2 = 5/18
        y = np.array([0])
        assert rps(_uniform_P(1), y) == pytest.approx(5.0 / 18.0, rel=1e-6)

    def test_uniform_draw(self):
        from cupcast.v2.model.validate import rps

        # cumP=[1/3,2/3,1], cumO=[0,1,1]
        # sum = (1/3)^2 + (1/3)^2 + 0 = 2/9; /2 = 1/9
        y = np.array([1])
        assert rps(_uniform_P(1), y) == pytest.approx(1.0 / 9.0, rel=1e-6)

    def test_uniform_away_win(self):
        from cupcast.v2.model.validate import rps

        # cumP=[1/3,2/3,1], cumO=[0,0,1]
        # sum = (1/3)^2 + (2/3)^2 + 0 = 5/9; /2 = 5/18
        y = np.array([2])
        assert rps(_uniform_P(1), y) == pytest.approx(5.0 / 18.0, rel=1e-6)

    def test_ordered_outcome_sensitivity(self):
        from cupcast.v2.model.validate import rps

        # Predicting home win strongly (0.9); away win is further in ordered scale than draw.
        P = np.array([[0.9, 0.05, 0.05]])
        # Draw (y=1): (0.9-0)^2 + (0.95-1)^2 = 0.81+0.0025 = 0.8125; /2 = 0.40625
        rps_draw = rps(P, np.array([1]))
        # Away win (y=2): (0.9-0)^2 + (0.95-0)^2 = 0.81+0.9025 = 1.7125; /2 = 0.85625
        rps_away = rps(P, np.array([2]))
        assert rps_away > rps_draw


# ---------------------------------------------------------------------------
# rolling_score
# ---------------------------------------------------------------------------


def _make_matches_all_home_wins(n: int = 20) -> pd.DataFrame:
    """Synthetic match table: all outcomes are home wins."""
    dates = pd.date_range("2023-01-01", periods=n, freq="ME")
    teams = ["A", "B", "C", "D"]
    return pd.DataFrame(
        [
            {
                "date": date,
                "home": teams[i % 4],
                "away": teams[(i + 1) % 4],
                "home_goals": 2,
                "away_goals": 0,
                "host_home": True,
            }
            for i, date in enumerate(dates)
        ]
    )


class TestRollingScore:
    def test_returns_expected_shape(self):
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins()

        def stub(train):
            def predict(home, away, host_home):
                return (1.0, 0.0, 0.0)

            return predict

        result = rolling_score(matches, stub, ["2023-06-01", "2024-01-01"])
        assert set(result.index) == {"model", "uniform"}
        assert list(result.columns) == ["n", "log_loss", "brier", "rps"]

    def test_perfect_model_beats_uniform_all_metrics(self):
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins()

        def stub(train):
            def predict(home, away, host_home):
                return (1.0, 0.0, 0.0)  # perfectly predicts home win

            return predict

        result = rolling_score(matches, stub, ["2023-06-01", "2024-01-01"])
        for metric in ("log_loss", "brier", "rps"):
            assert result.loc["model", metric] < result.loc["uniform", metric], (
                f"model should beat uniform on {metric}"
            )

    def test_perfect_model_zero_scores(self):
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins()

        def stub(train):
            def predict(home, away, host_home):
                return (1.0, 0.0, 0.0)

            return predict

        result = rolling_score(matches, stub, ["2023-06-01", "2024-01-01"])
        assert result.loc["model", "log_loss"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc["model", "brier"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc["model", "rps"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_forecaster_known_scores(self):
        from cupcast.v2.model.validate import rolling_score

        # All home wins; uniform predicts (1/3,1/3,1/3).
        matches = _make_matches_all_home_wins()

        def stub(train):
            def predict(home, away, host_home):
                return (1.0, 0.0, 0.0)

            return predict

        result = rolling_score(matches, stub, ["2023-06-01", "2024-01-01"])
        assert result.loc["uniform", "log_loss"] == pytest.approx(np.log(3), rel=1e-5)
        assert result.loc["uniform", "brier"] == pytest.approx(2.0 / 3.0, rel=1e-5)
        assert result.loc["uniform", "rps"] == pytest.approx(5.0 / 18.0, rel=1e-5)

    def test_fit_fn_receives_only_training_rows(self):
        """fit_fn must receive rows strictly before the cutoff."""
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins(20)
        received_sizes: list[int] = []

        def tracking_stub(train):
            received_sizes.append(len(train))

            def predict(home, away, host_home):
                return (1 / 3, 1 / 3, 1 / 3)

            return predict

        rolling_score(matches, tracking_stub, ["2023-06-01", "2024-01-01"])
        # Both folds have non-empty training sets; fit_fn should be called twice.
        assert len(received_sizes) == 2
        # Second fold trains on more data than first.
        assert received_sizes[1] > received_sizes[0]

    def test_empty_test_windows_skipped(self):
        """Folds where the test window has no matches are silently skipped."""
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins(5)

        def stub(train):
            def predict(home, away, host_home):
                return (1 / 3, 1 / 3, 1 / 3)

            return predict

        # No matches exist in 2030.
        result = rolling_score(matches, stub, ["2030-01-01", "2031-01-01"])
        assert list(result.columns) == ["n", "log_loss", "brier", "rps"]

    def test_n_counts_test_matches(self):
        from cupcast.v2.model.validate import rolling_score

        matches = _make_matches_all_home_wins(20)

        def stub(train):
            def predict(home, away, host_home):
                return (1.0, 0.0, 0.0)

            return predict

        result = rolling_score(matches, stub, ["2023-06-01", "2024-01-01"])
        # model and uniform see the same test set.
        assert result.loc["model", "n"] == result.loc["uniform", "n"]
        assert result.loc["model", "n"] > 0
