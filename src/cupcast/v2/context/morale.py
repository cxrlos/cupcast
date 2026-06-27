"""Morale signal stub — shadow mode only.

This module exposes the ``morale_signal`` interface so downstream callers can
request morale data and log the (team, as_of) pair. It always returns
``weight=0.0``; no morale covariate enters the probability model until an
ablation study proves marginal value. Real extraction will be wired in a
later task after that proof.
"""
from __future__ import annotations

from datetime import date


def morale_signal(team: str, as_of: date | None = None) -> dict:
    """Shadow-mode morale signal for *team* — always zero weight.

    Returns ``{"team": team, "value": 0.0, "weight": 0.0, "sources": []}``.
    Weight is permanently 0.0 in this release; the function is logged-only and
    never affects outcome probabilities.
    """
    return {"team": team, "value": 0.0, "weight": 0.0, "sources": []}
