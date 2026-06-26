"""WC2026 actual-results assembler and held-out v2 scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v2.fetch.endpoints import fetch_fixtures
from cupcast.v2.model import predict as _predict
from cupcast.v2.model.validate import brier, log_loss, rps

HOSTS: tuple[str, ...] = ("Mexico", "USA", "Canada")

_FINISHED: frozenset[str] = frozenset({"FT", "AET", "PEN"})

_RESULT_COLS = ["home", "away", "home_goals", "away_goals", "outcome", "stage"]


def _outcome(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def actual_wc2026_results(client) -> pd.DataFrame:
    """Return finished WC2026 match results sorted by date.

    Fetches league 1, season 2026, keeping only matches whose status is
    FT, AET, or PEN and whose goal counts are non-null.

    Returns
    -------
    DataFrame with columns ``home, away, home_goals, away_goals, outcome, stage``.
    ``outcome``: 0=home win, 1=draw, 2=away win.
    ``stage``: ``league.round`` string (e.g. "Group Stage - 1", "Round of 32").
    """
    fixtures = fetch_fixtures(client, 1, 2026)
    rows: list[dict] = []
    for f in fixtures:
        fix = f.get("fixture") or {}
        status = (fix.get("status") or {}).get("short")
        if status not in _FINISHED:
            continue
        goals = f.get("goals") or {}
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            continue
        teams = f.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if not home or not away:
            continue
        stage = ((f.get("league") or {}).get("round")) or ""
        date = pd.Timestamp(fix.get("date", ""), tz="UTC")
        rows.append(
            {
                "date": date,
                "home": home,
                "away": away,
                "home_goals": int(hg),
                "away_goals": int(ag),
                "outcome": _outcome(int(hg), int(ag)),
                "stage": stage,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_RESULT_COLS)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df[_RESULT_COLS]


def score_predictions(P: np.ndarray, y: np.ndarray) -> dict:
    """Score predictions against realized outcomes.

    Parameters
    ----------
    P:
        Probability triples, shape (n, 3) — (home win, draw, away win).
    y:
        Integer outcomes (0=home win, 1=draw, 2=away win), shape (n,).

    Returns
    -------
    Dict with keys ``n``, ``log_loss``, ``brier``, ``rps``.
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return {
        "n": len(y),
        "log_loss": log_loss(P, y),
        "brier": brier(P, y),
        "rps": rps(P, y),
    }


def held_out_v2(
    posterior,
    results: pd.DataFrame,
    hosts: tuple[str, ...] = HOSTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute held-out v2 predictions for WC2026 matches.

    Skips any match where either team is absent from ``posterior.teams``.
    Applies host advantage flags based on ``hosts``.

    Parameters
    ----------
    posterior:
        A fitted ``Posterior`` with a ``.teams`` attribute (tuple of team names).
    results:
        DataFrame from ``actual_wc2026_results``.
    hosts:
        Teams that receive host advantage (default: Mexico, USA, Canada).

    Returns
    -------
    ``(P, y)`` where P has shape (k, 3) and y has shape (k,), aligned.
    """
    known = set(posterior.teams)
    Ps: list[list[float]] = []
    ys: list[int] = []
    for _, row in results.iterrows():
        home, away = str(row["home"]), str(row["away"])
        if home not in known or away not in known:
            continue
        p = _predict.outcome_probs(
            posterior,
            home,
            away,
            host_home=home in hosts,
            host_away=away in hosts,
        )
        Ps.append(list(p))
        ys.append(int(row["outcome"]))

    if not Ps:
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    return np.array(Ps, dtype=float), np.array(ys, dtype=int)
