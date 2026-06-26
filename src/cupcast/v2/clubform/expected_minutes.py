"""Expected national-team minutes derived from real cached international lineups."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import pandas as pd


def national_team_appearances(
    lineups_by_fixture: dict[int, list[dict]],
    fixture_dates: dict[int, str],
    as_of: str | pd.Timestamp | None = None,
    half_life_days: float = 540,
) -> pd.DataFrame:
    """Per-player recency-weighted starts and sub appearances across international fixtures.

    Each fixture value in *lineups_by_fixture* is a list of two team blocks (one per side),
    each shaped like the API-Football ``/fixtures/lineups`` response.

    Returns a DataFrame with columns:
    ``player_id, team, w_starts, w_sub_apps, w_team_matches``.
    """
    if not lineups_by_fixture:
        return pd.DataFrame(
            columns=["player_id", "team", "w_starts", "w_sub_apps", "w_team_matches"]
        )

    dates = {fid: pd.Timestamp(d) for fid, d in fixture_dates.items()}
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else max(dates.values())
    decay = math.log(2) / half_life_days

    w_starts: dict[int, float] = defaultdict(float)
    w_sub_apps: dict[int, float] = defaultdict(float)
    player_team_counts: dict[int, Counter] = defaultdict(Counter)
    team_w_matches: dict[str, float] = defaultdict(float)

    for fixture_id, team_blocks in lineups_by_fixture.items():
        fixture_date = dates.get(fixture_id)
        if fixture_date is None:
            continue
        age_days = (as_of_ts - fixture_date).days
        w = math.exp(-decay * age_days)

        for block in team_blocks:
            team_name: str = block["team"]["name"]
            team_w_matches[team_name] += w

            for entry in block.get("startXI", []):
                pid: int = entry["player"]["id"]
                w_starts[pid] += w
                player_team_counts[pid][team_name] += 1

            for entry in block.get("substitutes", []):
                pid = entry["player"]["id"]
                w_sub_apps[pid] += w
                player_team_counts[pid][team_name] += 1

    all_pids = set(w_starts) | set(w_sub_apps)
    rows = []
    for pid in all_pids:
        team_name = player_team_counts[pid].most_common(1)[0][0]
        rows.append(
            {
                "player_id": pid,
                "team": team_name,
                "w_starts": w_starts[pid],
                "w_sub_apps": w_sub_apps[pid],
                "w_team_matches": team_w_matches[team_name],
            }
        )

    return pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)


def expected_minutes(
    appearances: pd.DataFrame,
    full_match: float = 90.0,
    sub_minutes_prior: float = 25.0,
) -> pd.DataFrame:
    """Start probability and expected tournament minutes per player.

    Consumes the output of :func:`national_team_appearances`.

    Returns a DataFrame with columns: ``player_id, start_prob, exp_minutes``.
    """
    df = appearances.copy()
    df["start_prob"] = (df["w_starts"] / df["w_team_matches"]).clip(0.0, 1.0)
    sub_rate = (df["w_sub_apps"] / df["w_team_matches"]).clip(upper=1.0)
    df["exp_minutes"] = (
        df["start_prob"] * full_match
        + (1.0 - df["start_prob"]) * sub_rate * sub_minutes_prior
    ).clip(0.0, 90.0)
    return df[["player_id", "start_prob", "exp_minutes"]].reset_index(drop=True)
