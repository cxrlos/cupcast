from __future__ import annotations

import pandas as pd

# v1 heuristic: a 4-3-3 baseline depth chart with starters chosen by caps
# within each position group. Goalkeepers are effectively never rotated at
# tournaments; outfield starters keep STARTER_SHARE of their slot's minutes
# and the bench splits the remainder in proportion to caps.
STARTERS_BY_POSITION = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}
STARTER_SHARE = 0.85
MATCH_MINUTES = 90


def _position_minutes(group: pd.DataFrame, n_starters: int) -> pd.Series:
    ranked = group.sort_values(["caps", "name"], ascending=[False, True])
    minutes = pd.Series(0.0, index=ranked.index)
    starters = ranked.index[:n_starters]
    bench = ranked.index[n_starters:]
    if group["position"].iloc[0] == "GK":
        if len(starters) > 0:
            minutes[starters[0]] = float(MATCH_MINUTES)
        return minutes
    minutes[starters] = MATCH_MINUTES * STARTER_SHARE
    pool = MATCH_MINUTES * len(starters) * (1 - STARTER_SHARE)
    if len(bench) > 0 and pool > 0:
        bench_caps = ranked.loc[bench, "caps"] + 1.0
        minutes[bench] = pool * bench_caps / bench_caps.sum()
    return minutes


def expected_minutes_per_match(squads: pd.DataFrame) -> pd.DataFrame:
    squads = squads.copy()
    squads["expected_minutes"] = 0.0
    for (_, position), group in squads.groupby(["team", "position"]):
        n_starters = min(STARTERS_BY_POSITION.get(position, 0), len(group))
        squads.loc[group.index, "expected_minutes"] = _position_minutes(group, n_starters)
    totals = squads.groupby("team")["expected_minutes"].transform("sum")
    squads["minutes_weight"] = squads["expected_minutes"] / totals
    return squads
