from __future__ import annotations

import pandas as pd

_FINAL_STATUSES = {"FT", "AET", "PEN"}


def build_match_table(fixtures_by_comp: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for competition, fixtures in fixtures_by_comp.items():
        for f in fixtures:
            status = (f.get("fixture", {}).get("status", {}) or {}).get("short")
            goals = f.get("goals", {}) or {}
            if (
                status not in _FINAL_STATUSES
                or goals.get("home") is None
                or goals.get("away") is None
            ):
                continue
            rows.append(
                {
                    "date": f["fixture"]["date"],
                    "home": f["teams"]["home"]["name"],
                    "away": f["teams"]["away"]["name"],
                    "home_goals": int(goals["home"]),
                    "away_goals": int(goals["away"]),
                    "competition": competition,
                    "season": int(f["league"]["season"]),
                    "status": status,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=["date", "home", "away", "home_goals", "away_goals",
                 "competition", "season", "status"],
    )
    table["date"] = pd.to_datetime(table["date"], utc=True)
    return table.sort_values("date").reset_index(drop=True)
