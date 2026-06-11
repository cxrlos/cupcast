from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0

# K-factors from the World Football Elo Ratings system (eloratings.net/about).
# The Nations League placement at qualifier level is a judgment call documented
# in the methodology paper.
K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0
K_QUALIFIER = 40.0
K_OTHER_TOURNAMENT = 30.0
K_FRIENDLY = 20.0


@dataclass(frozen=True)
class EloMatch:
    home: str
    away: str
    home_goals: int
    away_goals: int
    k: float
    neutral: bool = False


@dataclass(frozen=True)
class EloRecord:
    home_pre: float
    away_pre: float
    expected_home: float
    delta: float


def expected_score(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def goal_multiplier(goal_diff: int) -> float:
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8.0


def result_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


class EloRatings:
    def __init__(
        self,
        initial: float = DEFAULT_RATING,
        home_advantage: float = HOME_ADVANTAGE,
    ) -> None:
        self.initial = initial
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def play(self, match: EloMatch) -> EloRecord:
        home_pre = self.rating(match.home)
        away_pre = self.rating(match.away)
        diff = home_pre - away_pre
        if not match.neutral:
            diff += self.home_advantage
        expected = expected_score(diff)
        delta = (
            match.k
            * goal_multiplier(match.home_goals - match.away_goals)
            * (result_score(match.home_goals, match.away_goals) - expected)
        )
        self.ratings[match.home] = home_pre + delta
        self.ratings[match.away] = away_pre - delta
        return EloRecord(home_pre, away_pre, expected, delta)

    def run(self, matches: list[EloMatch]) -> list[EloRecord]:
        return [self.play(match) for match in matches]

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)
