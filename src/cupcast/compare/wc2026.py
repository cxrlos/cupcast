"""v1-vs-v2 held-out comparison on WC2026 results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v2.model.wc2026 import HOSTS, held_out_v2, score_predictions

V1_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def held_out_v1(
    v1_fit,
    results: pd.DataFrame,
    hosts: tuple[str, ...] = HOSTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute held-out v1 predictions for WC2026 matches.

    Maps API-Football team names to v1 names via ``V1_NAME_MAP``. Host flags
    are checked against the original API-Football names; ``outcome_probs``
    receives the mapped v1 names. Rows where either mapped name is absent from
    ``v1_fit.teams`` are skipped.

    Parameters
    ----------
    v1_fit:
        A ``DixonColesFit`` with ``.teams`` and
        ``.outcome_probs(home, away, host_home, host_away)``.
    results:
        DataFrame from ``actual_wc2026_results``.
    hosts:
        Teams that receive host advantage, checked against original API-Football names.

    Returns
    -------
    ``(P, y)`` where P has shape (k, 3) and y has shape (k,), aligned.
    """
    known = set(v1_fit.teams)
    Ps: list[list[float]] = []
    ys: list[int] = []
    for _, row in results.iterrows():
        home, away = str(row["home"]), str(row["away"])
        home_v1 = V1_NAME_MAP.get(home, home)
        away_v1 = V1_NAME_MAP.get(away, away)
        if home_v1 not in known or away_v1 not in known:
            continue
        p = v1_fit.outcome_probs(
            home_v1,
            away_v1,
            host_home=home in hosts,
            host_away=away in hosts,
        )
        Ps.append(list(p))
        ys.append(int(row["outcome"]))

    if not Ps:
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    return np.array(Ps, dtype=float), np.array(ys, dtype=int)


def compare_on_common(
    results: pd.DataFrame,
    v2_posterior,
    v1_fit,
    hosts: tuple[str, ...] = HOSTS,
) -> pd.DataFrame:
    """Score v2, v1, and uniform on the matches scorable by BOTH models.

    Parameters
    ----------
    results:
        DataFrame from ``actual_wc2026_results``.
    v2_posterior:
        Fitted ``Posterior`` for the v2 model.
    v1_fit:
        Fitted ``DixonColesFit`` for the v1 model.
    hosts:
        Teams that receive host advantage.

    Returns
    -------
    DataFrame with columns ``forecaster, n, log_loss, brier, rps``; one row
    each for ``v2``, ``v1``, and ``uniform``.
    """
    v2_known = set(v2_posterior.teams)
    v1_known = set(v1_fit.teams)

    mask = []
    for _, row in results.iterrows():
        home, away = str(row["home"]), str(row["away"])
        home_v1 = V1_NAME_MAP.get(home, home)
        away_v1 = V1_NAME_MAP.get(away, away)
        mask.append(
            home in v2_known
            and away in v2_known
            and home_v1 in v1_known
            and away_v1 in v1_known
        )

    common = results[mask].reset_index(drop=True)

    P_v2, y = held_out_v2(v2_posterior, common, hosts=hosts)
    P_v1, _ = held_out_v1(v1_fit, common, hosts=hosts)
    n = len(y)
    P_uniform = np.full((n, 3), 1.0 / 3.0)

    rows = [
        {"forecaster": name, **score_predictions(P, y)}
        for name, P in (("v2", P_v2), ("v1", P_v1), ("uniform", P_uniform))
    ]
    return pd.DataFrame(rows)
