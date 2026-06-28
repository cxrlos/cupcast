"""Penalty-specific shootout quality composite for cupcast.v2.clubform."""

from __future__ import annotations

import pandas as pd

from cupcast.v2.clubform.player_quality import canonical_position
from cupcast.v2.fetch.endpoints import fetch_players


def parse_penalty_stats(
    players_response: list[dict],
    league_id: int,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Extract penalty-specific stats from an API-Football /players response.

    Uses the first statistics block per entry. Players below *min_minutes* are
    dropped. Null penalty values are treated as 0.

    Returns columns: ``player_id, name, position, minutes, pen_scored,
    pen_missed, pen_saved``.
    """
    rows: list[dict] = []
    for entry in players_response:
        player = entry["player"]
        stats = entry["statistics"][0]

        games = stats["games"]
        minutes = float(games.get("minutes") or 0)
        if minutes < min_minutes:
            continue

        position = canonical_position(games.get("position"))
        penalty = stats.get("penalty") or {}

        rows.append(
            {
                "player_id": player["id"],
                "name": player["name"],
                "position": position,
                "minutes": minutes,
                "pen_scored": int(penalty.get("scored") or 0),
                "pen_missed": int(penalty.get("missed") or 0),
                "pen_saved": int(penalty.get("saved") or 0),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "player_id", "name", "position", "minutes",
            "pen_scored", "pen_missed", "pen_saved",
        ],
    )


def _zscore(s: pd.Series) -> pd.Series:
    """Population z-score (ddof=0); empty, single, or zero-variance series → 0."""
    if len(s) == 0:
        return s.copy()
    if len(s) == 1:
        return pd.Series(0.0, index=s.index)
    std = float(s.std(ddof=0))
    if std == 0.0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def team_shootout_z(
    penalty_by_league: dict[int, list[dict]],
    exp_minutes: pd.DataFrame,
    strength_index: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Per-team penalty-shootout z-score from penalty stats and expected minutes.

    *penalty_by_league* maps league_id → raw /players responses. *exp_minutes*
    must have columns ``player_id, team, exp_minutes``.

    For each team:
    - ``keeper_signal``: exp-minutes-weighted mean ``pen_saved`` over matched GKs.
    - ``taker_signal``: exp-minutes-weighted penalty-conversion rate over outfield
      players with at least one attempt; teams with no such players → neutral 0.

    Both signals are z-scored across teams with data, summed to ``shootout_raw``,
    then z-scored again to ``shootout_z``. Teams with no matched penalty data → 0.

    Returns columns: ``team, shootout_z``.
    """
    frames: list[pd.DataFrame] = []
    for league_id, responses in penalty_by_league.items():
        parsed = parse_penalty_stats(responses, league_id)
        if not parsed.empty:
            frames.append(parsed)

    if frames:
        all_pen = (
            pd.concat(frames, ignore_index=True)
            .sort_values("minutes", ascending=False)
            .drop_duplicates(subset="player_id", keep="first")
            .reset_index(drop=True)
        )
        pen_cols = all_pen[["player_id", "position", "pen_scored", "pen_missed", "pen_saved"]]
    else:
        pen_cols = pd.DataFrame(
            columns=["player_id", "position", "pen_scored", "pen_missed", "pen_saved"]
        )

    merged = exp_minutes.merge(pen_cols, on="player_id", how="left")

    team_rows: list[dict] = []
    for team, grp in merged.groupby("team"):
        matched = grp.dropna(subset=["position"])
        has_data = not matched.empty

        if has_data:
            gk = matched[matched["position"] == "Goalkeeper"]
            total_gk_em = float(gk["exp_minutes"].sum())
            keeper_signal = (
                float((gk["pen_saved"] * gk["exp_minutes"]).sum() / total_gk_em)
                if total_gk_em > 0
                else 0.0
            )

            outfield = matched[matched["position"] != "Goalkeeper"].copy()
            outfield["pen_attempts"] = outfield["pen_scored"] + outfield["pen_missed"]
            takers = outfield[outfield["pen_attempts"] >= 1]
            total_t_em = float(takers["exp_minutes"].sum())
            taker_signal = (
                float(
                    (takers["pen_scored"] / takers["pen_attempts"] * takers["exp_minutes"]).sum()
                    / total_t_em
                )
                if total_t_em > 0
                else 0.0
            )
        else:
            keeper_signal = 0.0
            taker_signal = 0.0

        team_rows.append(
            {
                "team": team,
                "keeper_signal": keeper_signal,
                "taker_signal": taker_signal,
                "has_data": has_data,
            }
        )

    if not team_rows:
        return pd.DataFrame(columns=["team", "shootout_z"])

    df = pd.DataFrame(team_rows)
    data_mask = df["has_data"]

    with_data = df[data_mask].copy()
    without_data = df[~data_mask][["team"]].copy()
    without_data["shootout_z"] = 0.0

    parts: list[pd.DataFrame] = [without_data]

    if not with_data.empty:
        with_data["keeper_z"] = _zscore(with_data["keeper_signal"])
        with_data["taker_z"] = _zscore(with_data["taker_signal"])
        with_data["shootout_raw"] = with_data["keeper_z"] + with_data["taker_z"]
        with_data["shootout_z"] = _zscore(with_data["shootout_raw"])
        parts.append(with_data[["team", "shootout_z"]])

    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def build_shootout(
    client,
    club_league_seasons: list[tuple[int, int]],
    intl_comp_seasons: list[tuple[int, int]],
) -> dict[str, float]:
    """Fetch penalty stats and return per-team shootout z-scores.

    Uses :func:`~cupcast.v2.clubform.composite.assemble_expected_minutes` for
    national-team expected minutes. Returns a dict ready to pass as ``gk_z``
    to the knockout simulator.
    """
    from cupcast.v2.clubform.composite import assemble_expected_minutes  # noqa: PLC0415

    penalty_by_league: dict[int, list[dict]] = {}
    for league_id, season in club_league_seasons:
        responses = fetch_players(client, league_id, season)
        penalty_by_league.setdefault(league_id, []).extend(responses)

    exp_minutes = assemble_expected_minutes(client, intl_comp_seasons)
    result = team_shootout_z(penalty_by_league, exp_minutes)
    return dict(zip(result["team"], result["shootout_z"], strict=True))
