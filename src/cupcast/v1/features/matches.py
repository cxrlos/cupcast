from __future__ import annotations

import pandas as pd

from cupcast.v1.fetch.martj42 import load_results
from cupcast.v1.ratings.elo import (
    K_CONTINENTAL,
    K_FRIENDLY,
    K_OTHER_TOURNAMENT,
    K_QUALIFIER,
    K_WORLD_CUP,
)

# Continental championship finals tournaments, per the eloratings.net K hierarchy.
CONTINENTAL_FINALS = {
    "uefa euro",
    "copa américa",
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "oceania nations cup",
    "confederations cup",
}


def k_factor(tournament: str) -> float:
    name = tournament.casefold()
    if name == "fifa world cup":
        return K_WORLD_CUP
    if name in CONTINENTAL_FINALS:
        return K_CONTINENTAL
    if "qualification" in name or "nations league" in name:
        return K_QUALIFIER
    if name == "friendly":
        return K_FRIENDLY
    return K_OTHER_TOURNAMENT


def restrict_to_fifa_pool(matches: pd.DataFrame) -> pd.DataFrame:
    # martj42 includes non-FIFA internationals (CONIFA and the like). Every FIFA
    # member appears in World Cup qualification or a Nations League within any
    # multi-year window, so that participation defines the pool.
    anchored = matches[matches["k"] >= K_QUALIFIER]
    pool = set(anchored["home"]) | set(anchored["away"])
    return matches[matches["home"].isin(pool) & matches["away"].isin(pool)]


def build_match_table(since: str | None = None, fifa_pool: bool = True) -> pd.DataFrame:
    matches = load_results(since=since)
    matches["k"] = matches["tournament"].map(k_factor)
    matches["friendly"] = matches["tournament"].str.casefold() == "friendly"
    if fifa_pool:
        matches = restrict_to_fifa_pool(matches)
    return matches.sort_values("date", ignore_index=True)
