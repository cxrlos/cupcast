"""International match table assembled from cached API-Football fixtures."""

from __future__ import annotations

import re

import pandas as pd

from cupcast.v2.fetch import endpoints

# Competition IDs that are single-host neutral-site tournaments.
_NEUTRAL_COMPS: frozenset[int] = frozenset({1, 4, 6, 7, 9, 22})

_FINISHED: frozenset[str] = frozenset({"FT", "AET", "PEN"})

# The Friendlies competition lumps senior, youth (U15-U23), B, and women's
# internationals together. Only senior men's national teams belong in the model.
_NON_SENIOR = re.compile(r"\bU-?\d{2}\b| B$| W$|Women|Olympic", re.IGNORECASE)


def _is_senior_men(name: str) -> bool:
    return not _NON_SENIOR.search(name or "")


def assemble_internationals(
    client,
    comp_seasons: list[tuple[int, int]],
    cutoff: str | None = None,
) -> pd.DataFrame:
    """Load finished international fixtures and return a clean match table.

    Parameters
    ----------
    client:
        An ``ApiFootballClient`` (or stub) supplying ``get_response``.
    comp_seasons:
        Pairs of (league_id, season) to fetch.
    cutoff:
        ISO date string; matches on or after this date are excluded (holds out
        WC2026 when set to the tournament start date).

    Returns
    -------
    DataFrame with columns:
        date, home, away, home_goals, away_goals, competition,
        neutral, host_home, period
    """
    rows: list[dict] = []
    for league_id, season in comp_seasons:
        for fixture in endpoints.fetch_fixtures(client, league_id, season):
            fix_meta = fixture.get("fixture") or {}
            status = (fix_meta.get("status") or {}).get("short")
            if status not in _FINISHED:
                continue
            goals = fixture.get("goals") or {}
            if goals.get("home") is None or goals.get("away") is None:
                continue
            teams = fixture.get("teams") or {}
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            if not home or not away:
                continue
            if not (_is_senior_men(home) and _is_senior_men(away)):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(fix_meta.get("date", ""), tz="UTC"),
                    "home": home,
                    "away": away,
                    "home_goals": int(goals["home"]),
                    "away_goals": int(goals["away"]),
                    "competition": league_id,
                    "neutral": league_id in _NEUTRAL_COMPS,
                }
            )

    _empty_cols = [
        "date",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "competition",
        "neutral",
        "host_home",
        "period",
    ]
    if not rows:
        return pd.DataFrame(columns=_empty_cols)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Keep only FIFA national teams: those appearing in at least one official
    # (non-friendly) competition. Drops clubs and non-FIFA selections that only
    # surface in friendlies (e.g. "FC Urartu", "Basque Country", "Catalonia").
    official = set(df.loc[df["competition"] != 10, "home"]) | set(
        df.loc[df["competition"] != 10, "away"]
    )
    df = df[df["home"].isin(official) & df["away"].isin(official)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=_empty_cols)

    df["host_home"] = ~df["neutral"]

    min_year = int(df["date"].dt.year.min())
    df["period"] = (df["date"].dt.year - min_year) * 4 + (df["date"].dt.month - 1) // 3

    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
        df = df[df["date"] < cutoff_ts].reset_index(drop=True)

    return df[_empty_cols]
