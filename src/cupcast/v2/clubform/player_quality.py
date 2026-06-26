"""Position-aware player-quality parser over API-Football season stats."""

from __future__ import annotations

import pandas as pd

# API-Football emits "Forward" as a synonym for "Attacker"; canonicalize so
# within-position z-scoring treats them as one group.
_CANONICAL_POSITION = {"Forward": "Attacker"}


def canonical_position(position: str | None) -> str:
	"""Return canonical position string, mapping Forward->Attacker."""
	pos = (position or "").strip()
	return _CANONICAL_POSITION.get(pos, pos)


def _coerce(val) -> float:
    """Return 0.0 for None/null stat values, otherwise cast to float."""
    return 0.0 if val is None else float(val)


def parse_player_stats(
    players_response: list[dict],
    league_id: int,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Parse API-Football /players response into per-player quality metrics.

    Uses the FIRST statistics block for each player entry. Players with fewer
    than *min_minutes* are dropped.

    Returns a DataFrame with columns: ``player_id, name, position, minutes,
    rating, att_raw, def_raw, gk_raw, save_rate``.
    """
    rows: list[dict] = []

    for entry in players_response:
        player = entry["player"]
        stats = entry["statistics"][0]

        games = stats["games"]
        minutes = _coerce(games.get("minutes"))
        if minutes < min_minutes:
            continue

        position: str = canonical_position(games.get("position"))
        rating_raw = games.get("rating")
        rating = 0.0 if rating_raw is None else float(rating_raw)

        goals = stats.get("goals", {})
        shots = stats.get("shots", {})
        passes = stats.get("passes", {})
        tackles = stats.get("tackles", {})
        duels = stats.get("duels", {})

        goals_total = _coerce(goals.get("total"))
        goals_assists = _coerce(goals.get("assists"))
        goals_saves = _coerce(goals.get("saves"))
        goals_conceded = _coerce(goals.get("conceded"))
        shots_on = _coerce(shots.get("on"))
        passes_key = _coerce(passes.get("key"))
        tackles_total = _coerce(tackles.get("total"))
        tackles_blocks = _coerce(tackles.get("blocks"))
        tackles_interceptions = _coerce(tackles.get("interceptions"))
        duels_won = _coerce(duels.get("won"))

        m90 = minutes / 90.0

        att_raw = (
            (goals_total + goals_assists) / m90
            + 0.5 * shots_on / m90
            + 0.3 * passes_key / m90
        )
        def_raw = (
            (tackles_total + tackles_interceptions + tackles_blocks + duels_won) / m90
        )

        if position == "Goalkeeper":
            gk_raw = goals_saves / m90
            save_denom = goals_saves + goals_conceded
            save_rate = goals_saves / save_denom if save_denom > 0 else float("nan")
        else:
            gk_raw = float("nan")
            save_rate = float("nan")

        rows.append(
            {
                "player_id": player["id"],
                "name": player["name"],
                "position": position,
                "minutes": minutes,
                "rating": rating,
                "att_raw": att_raw,
                "def_raw": def_raw,
                "gk_raw": gk_raw,
                "save_rate": save_rate,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "player_id", "name", "position", "minutes", "rating",
            "att_raw", "def_raw", "gk_raw", "save_rate",
        ],
    )


def apply_league_strength(
    quality: pd.DataFrame,
    league_id: int,
    strength_index: dict[int, float],
) -> pd.DataFrame:
    """Multiply ``att_raw`` and ``def_raw`` by the league-strength factor.

    ``gk_raw`` and ``save_rate`` are left unscaled. Returns a copy.
    """
    factor = strength_index.get(league_id, 1.0)
    out = quality.copy()
    out["att_raw"] = out["att_raw"] * factor
    out["def_raw"] = out["def_raw"] * factor
    return out


def _zscore(series: pd.Series) -> pd.Series:
    """Population z-score (ddof=0) within a group; single/zero-var groups → 0."""
    if series.isna().all():
        return pd.Series(float("nan"), index=series.index)
    if len(series) <= 1:
        return pd.Series(0.0, index=series.index)
    std = series.std(ddof=0)
    if std == 0.0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def zscore_within_position(quality: pd.DataFrame) -> pd.DataFrame:
    """Add ``att_z``, ``def_z``, ``gk_z`` — each z-scored within its position group.

    ``gk_z`` is meaningful only for Goalkeepers; non-GK rows carry NaN there.
    Single-player or zero-variance groups receive z = 0.
    """
    out = quality.copy()
    out["att_z"] = out.groupby("position")["att_raw"].transform(_zscore)
    out["def_z"] = out.groupby("position")["def_raw"].transform(_zscore)
    out["gk_z"] = out.groupby("position")["gk_raw"].transform(_zscore)
    return out


def apply_understat_xg(
    quality: pd.DataFrame,
    understat: pd.DataFrame | None,
) -> pd.DataFrame:
    """Override ``att_raw`` with Understat npxG+xA per-90 where a player is matched.

    Matching is done by accent/case-folded name. No-op if *understat* is empty
    or None. Returns a copy of *quality*.
    """
    if understat is None or understat.empty:
        return quality

    from cupcast.v2.clubform.league_strength import normalize_club_name  # noqa: PLC0415

    understat_map: dict[str, float] = (
        understat.set_index(understat["player"].map(normalize_club_name))[
            "quality_per90"
        ].to_dict()
    )

    out = quality.copy()
    norm_names = out["name"].map(normalize_club_name)
    for idx, norm in norm_names.items():
        if norm in understat_map:
            out.at[idx, "att_raw"] = float(understat_map[norm])
    return out
