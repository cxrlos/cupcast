from __future__ import annotations

import pandas as pd

from cupcast.v1.ratings.elo import EloMatch, EloRatings


def elo_history(matches: pd.DataFrame, initial: float = 1500.0) -> tuple[dict, pd.DataFrame]:
    engine = EloRatings(initial=initial)
    rows = []
    for match in matches.sort_values("date").itertuples():
        record = engine.play(
            EloMatch(
                home=match.home,
                away=match.away,
                home_goals=match.home_goals,
                away_goals=match.away_goals,
                k=match.k,
                neutral=bool(match.neutral),
            )
        )
        rows.append(
            {
                "date": match.date,
                "home": match.home,
                "away": match.away,
                "home_elo_pre": record.home_pre,
                "away_elo_pre": record.away_pre,
                "expected_home": record.expected_home,
            }
        )
    return engine.snapshot(), pd.DataFrame(rows)
