import pytest

from cupcast.v1.ratings.elo import (
    EloMatch,
    EloRatings,
    expected_score,
    goal_multiplier,
)


def test_expected_score_is_symmetric():
    assert expected_score(0) == 0.5
    assert expected_score(120) + expected_score(-120) == pytest.approx(1.0)


def test_goal_multiplier_steps():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(-2) == 1.5
    assert goal_multiplier(3) == 1.75
    assert goal_multiplier(5) == 1.75 + 2 / 8


def test_neutral_win_between_equals_moves_half_k():
    elo = EloRatings()
    record = elo.play(EloMatch("Spain", "Argentina", 1, 0, k=40.0, neutral=True))
    assert record.delta == pytest.approx(20.0)
    assert elo.rating("Spain") == pytest.approx(1520.0)
    assert elo.rating("Argentina") == pytest.approx(1480.0)


def test_three_goal_win_uses_multiplier():
    elo = EloRatings()
    record = elo.play(EloMatch("Brazil", "Bolivia", 3, 0, k=40.0, neutral=True))
    assert record.delta == pytest.approx(40.0 * 1.75 * 0.5)


def test_home_team_loses_rating_on_home_draw():
    elo = EloRatings()
    record = elo.play(EloMatch("France", "Germany", 1, 1, k=40.0, neutral=False))
    assert record.expected_home > 0.5
    assert record.delta < 0
    assert elo.rating("France") < 1500.0 < elo.rating("Germany")


def test_zero_sum_and_unknown_teams_start_at_initial():
    elo = EloRatings(initial=1600.0)
    elo.run(
        [
            EloMatch("Japan", "Korea Republic", 2, 1, k=30.0),
            EloMatch("Korea Republic", "Japan", 0, 0, k=30.0),
        ]
    )
    total = sum(elo.snapshot().values())
    assert total == pytest.approx(2 * 1600.0)
