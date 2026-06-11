import numpy as np
import pytest

from cupcast.validate.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    ranked_probability_score,
)


def test_log_loss_hand_computed():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    outcome = np.array([0, 2])
    expected = -(np.log(0.5) + np.log(0.7)) / 2
    assert log_loss(probs, outcome) == pytest.approx(expected)


def test_brier_hand_computed():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert brier_score(probs, np.array([0])) == 0.0
    probs = np.array([[0.5, 0.25, 0.25]])
    assert brier_score(probs, np.array([0])) == pytest.approx(
        (0.5 - 1) ** 2 + 0.25**2 + 0.25**2
    )


def test_rps_rewards_ordered_closeness():
    outcome = np.array([0])
    near_miss = np.array([[0.0, 1.0, 0.0]])  # mass on draw
    far_miss = np.array([[0.0, 0.0, 1.0]])  # mass on away win
    assert ranked_probability_score(near_miss, outcome) < ranked_probability_score(
        far_miss, outcome
    )
    perfect = np.array([[1.0, 0.0, 0.0]])
    assert ranked_probability_score(perfect, outcome) == 0.0


def test_calibration_table_recovers_rates():
    rng = np.random.default_rng(0)
    n = 20_000
    p_home = rng.uniform(0.1, 0.8, n)
    rest = 1 - p_home
    probs = np.column_stack([p_home, rest / 2, rest / 2])
    outcome = np.array(
        [rng.choice(3, p=row) for row in probs]
    )
    table = calibration_table(probs, outcome, bins=5)
    assert ((table["observed_rate"] - table["mean_predicted"]).abs() < 0.03).all()
