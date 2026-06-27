"""Bounded, audited context adjustments on top of predict.score_matrix.

Each covariate maps to a capped log-rate nudge; exp(sum) gives the per-side
rate multiplier passed to score_matrix as rate_scale_home / rate_scale_away.
A provenance list records every non-zero effect for post-match auditing.

Cap rationale (hand-set, conservative):
  host    ≤ +0.12  — venue familiarity / crowd; ~12 % scoring rate lift at most
  travel  ≤ −0.10  — long-haul fatigue (≥ 3000 km saturates)
  rest    ≤ −0.08  — very short turnaround (0 days = max penalty, ≥ 4 days = 0)
  altitude≤ −0.10  — non-adapted side at high elevation (host is exempt)
"""

from __future__ import annotations

import math

import numpy as np

from cupcast.v2.model.fit import Posterior
from cupcast.v2.model.predict import score_matrix

CAPS: dict[str, float] = {
    "host": 0.12,
    "travel": 0.10,
    "rest": 0.08,
    "altitude": 0.10,
}


def covariate_log_effects(ctx: dict) -> dict[str, float]:
    """Convert one side's match_context dict into capped log-rate effects.

    Each returned value lies within ±CAPS[covariate] by construction;
    a defensive clip is applied on every path.
    """
    def _clip(val: float, key: str) -> float:
        return float(np.clip(val, -CAPS[key], CAPS[key]))

    host_raw = CAPS["host"] if ctx["is_host"] else 0.0
    host = _clip(host_raw, "host")

    travel_km = ctx["travel_km"]
    travel = _clip(-CAPS["travel"] * min(travel_km / 3000.0, 1.0), "travel")

    rest_days = ctx["rest_days"]
    if rest_days is not None and rest_days < 4:
        rest_raw = -CAPS["rest"] * (4 - rest_days) / 4.0
    else:
        rest_raw = 0.0
    rest = _clip(rest_raw, "rest")

    alt = ctx["altitude_m"]
    if alt is not None and alt > 1000 and not ctx["is_host"]:
        alt_raw = -CAPS["altitude"] * min((alt - 1000) / 1500.0, 1.0)
    else:
        alt_raw = 0.0
    altitude = _clip(alt_raw, "altitude")

    return {"host": host, "travel": travel, "rest": rest, "altitude": altitude}


def rate_multipliers(
    ctx_home: dict, ctx_away: dict
) -> tuple[float, float, list[dict]]:
    """Return (mult_home, mult_away, provenance) from two covariate dicts.

    mult_* = exp(sum of capped log effects for that side).
    provenance lists one record per non-zero effect:
    {side, covariate, raw, log_effect}.
    """
    _raw = {
        "host": lambda ctx: ctx["is_host"],
        "travel": lambda ctx: ctx["travel_km"],
        "rest": lambda ctx: ctx["rest_days"],
        "altitude": lambda ctx: ctx["altitude_m"],
    }

    home_fx = covariate_log_effects(ctx_home)
    away_fx = covariate_log_effects(ctx_away)

    provenance: list[dict] = []
    for covariate, effect in home_fx.items():
        if effect != 0.0:
            provenance.append({
                "side": "home",
                "covariate": covariate,
                "raw": _raw[covariate](ctx_home),
                "log_effect": effect,
            })
    for covariate, effect in away_fx.items():
        if effect != 0.0:
            provenance.append({
                "side": "away",
                "covariate": covariate,
                "raw": _raw[covariate](ctx_away),
                "log_effect": effect,
            })

    mult_home = math.exp(sum(home_fx.values()))
    mult_away = math.exp(sum(away_fx.values()))
    return mult_home, mult_away, provenance


def adjusted_outcome_probs(
    posterior: Posterior,
    home: str,
    away: str,
    ctx_home: dict,
    ctx_away: dict,
    host_home: bool = False,
    host_away: bool = False,
) -> tuple[tuple[float, float, float], list[dict]]:
    """Outcome probs with context multipliers applied, plus a provenance audit.

    Returns ((p_home_win, p_draw, p_away_win), provenance).
    When both context dicts carry zero effects the result is identical to
    predict.outcome_probs with the same host flags.
    """
    mh, ma, prov = rate_multipliers(ctx_home, ctx_away)
    matrix = score_matrix(
        posterior, home, away,
        host_home=host_home, host_away=host_away,
        rate_scale_home=mh, rate_scale_away=ma,
    )
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = 1.0 - p_home - p_draw
    return (p_home, p_draw, p_away), prov
