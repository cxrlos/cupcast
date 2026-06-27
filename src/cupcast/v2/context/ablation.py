"""Pre/post-context ablation on held-out WC2026 results.

``context_held_out`` builds three aligned arrays — base probabilities,
context-adjusted probabilities, and realized outcomes — for every finished
WC2026 fixture where both teams appear in the posterior.

Host-advantage accounting
-------------------------
Host advantage must not be double-counted.  Both the base ``outcome_probs``
call and the ``adjusted_outcome_probs`` call receive ``host_home=False,
host_away=False`` at the posterior level.  The context layer's ``is_host``
covariate (set by ``match_context``) is the sole source of host advantage.
This makes ``P_base`` vs ``P_ctx`` a clean model-only vs model+context
comparison.
"""
from __future__ import annotations

import numpy as np

from cupcast.v2.context.adjust import adjusted_outcome_probs
from cupcast.v2.context.covariates import match_context
from cupcast.v2.model import predict
from cupcast.v2.model.fit import Posterior
from cupcast.v2.model.wc2026 import HOSTS

_FINISHED: frozenset[str] = frozenset({"FT", "AET", "PEN"})
_GROUP_STAGE_PREFIX = "Group Stage"


def _outcome(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _league_round(fixture: dict) -> str | None:
    return (fixture.get("league") or {}).get("round")


def context_held_out(
    posterior: Posterior,
    fixtures: list[dict],
    hosts: tuple[str, ...] = HOSTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned (P_base, P_ctx, y) for finished WC2026 fixtures.

    Parameters
    ----------
    posterior:
        Fitted posterior; fixtures where either team is absent are skipped.
    fixtures:
        All WC2026 fixtures (finished + scheduled), as returned by
        ``fetch_fixtures``.  Group-stage fixtures are extracted internally
        to supply travel/rest schedule context via ``match_context``.
    hosts:
        Host nations; consumed only by the context layer (``is_host``),
        never passed to the base posterior call.

    Returns
    -------
    ``(P_base, P_ctx, y)`` — each P has shape (k, 3) with columns
    (p_home_win, p_draw, p_away_win); y has shape (k,) with values
    0=home win, 1=draw, 2=away win.
    """
    known = set(posterior.teams)
    group_fixtures = [
        f for f in fixtures
        if (_league_round(f) or "").startswith(_GROUP_STAGE_PREFIX)
    ]

    P_base_rows: list[list[float]] = []
    P_ctx_rows: list[list[float]] = []
    ys: list[int] = []

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
        if home not in known or away not in known:
            continue

        outcome = _outcome(int(hg), int(ag))
        p_base = predict.outcome_probs(
            posterior, home, away, host_home=False, host_away=False
        )
        ctx_home = match_context(group_fixtures, home, f)
        ctx_away = match_context(group_fixtures, away, f)
        p_ctx, _ = adjusted_outcome_probs(
            posterior, home, away, ctx_home, ctx_away,
            host_home=False, host_away=False,
        )

        P_base_rows.append(list(p_base))
        P_ctx_rows.append(list(p_ctx))
        ys.append(outcome)

    if not P_base_rows:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=int),
        )

    return (
        np.array(P_base_rows, dtype=float),
        np.array(P_ctx_rows, dtype=float),
        np.array(ys, dtype=int),
    )
