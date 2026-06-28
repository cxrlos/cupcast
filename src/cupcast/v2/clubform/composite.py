"""Squad-composite assembler and style profile for cupcast.v2.clubform."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cupcast.v2.clubform.expected_minutes import expected_minutes as _expected_minutes
from cupcast.v2.clubform.expected_minutes import national_team_appearances
from cupcast.v2.clubform.player_quality import (
    apply_league_strength,
    parse_player_stats,
    zscore_within_position,
)
from cupcast.v2.fetch.endpoints import fetch_fixtures, fetch_lineups, fetch_players, fixture_ids

_ATTACK_POSITIONS = {"Attacker", "Midfielder"}
_DEFENSE_POSITIONS = {"Defender", "Midfielder"}


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    v = values[mask]
    w = weights[mask]
    return float((v * w).sum() / w.sum())


def assemble_player_quality(
    player_responses_by_league: dict[int, list[dict]],
    strength_index: dict[int, float],
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Merge per-league player stats into a single pool with global z-scores.

    For each league: parse → apply league strength, then concatenate. When a
    player_id appears in multiple leagues the row with the most minutes is kept.
    Z-scores are computed within position groups over the combined global pool.

    Returns columns: ``player_id, name, position, minutes, att_z, def_z, gk_z, save_rate``.
    """
    _empty = pd.DataFrame(
        columns=["player_id", "name", "position", "minutes", "att_z", "def_z", "gk_z", "save_rate"]
    )

    frames: list[pd.DataFrame] = []
    for league_id, responses in player_responses_by_league.items():
        parsed = parse_player_stats(responses, league_id, min_minutes)
        if parsed.empty:
            continue
        frames.append(apply_league_strength(parsed, league_id, strength_index))

    if not frames:
        return _empty

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("minutes", ascending=False)
        .drop_duplicates(subset="player_id", keep="first")
        .reset_index(drop=True)
    )

    scored = zscore_within_position(combined)
    cols = ["player_id", "name", "position", "minutes", "att_z", "def_z", "gk_z", "save_rate"]
    return scored[cols]


def squad_profiles(
    player_quality: pd.DataFrame,
    exp_minutes: pd.DataFrame,
) -> pd.DataFrame:
    """Minutes-weighted squad composites per national team.

    Parameters
    ----------
    player_quality:
        Output of :func:`assemble_player_quality`.
    exp_minutes:
        Columns: ``player_id, team, exp_minutes``.

    Returns
    -------
    DataFrame with columns: ``team, attack, defense, gk, coverage, n_players``.
    Unmatched players (no row in player_quality) are imputed at the 20th-percentile
    att_z / def_z of the matched pool so missing data lowers the composite.
    """
    if player_quality.empty:
        p20_att = 0.0
        p20_def = 0.0
    else:
        p20_att = float(np.percentile(player_quality["att_z"].dropna(), 20))
        p20_def = float(np.percentile(player_quality["def_z"].dropna(), 20))

    # Precompute save_rate z-scores within the global GK pool for the GK fallback.
    save_rate_z_map: dict[int, float] = {}
    gk_rows = player_quality[player_quality["position"] == "Goalkeeper"].copy()
    if not gk_rows.empty:
        sr = gk_rows["save_rate"]
        n_valid = int(sr.notna().sum())
        if n_valid > 1:
            std = float(sr.std(ddof=0))
            mean = float(sr.mean())
            gk_rows["_sr_z"] = (sr - mean) / std if std > 0 else 0.0
        else:
            # Single or zero valid save_rate: z-score is undefined → 0 where present, NaN otherwise.
            gk_rows["_sr_z"] = sr.apply(lambda v: 0.0 if pd.notna(v) else float("nan"))
        save_rate_z_map = dict(zip(gk_rows["player_id"], gk_rows["_sr_z"], strict=True))

    # Left-join to identify unmatched players.
    merged = exp_minutes.merge(
        player_quality[["player_id", "position", "att_z", "def_z", "gk_z", "save_rate"]],
        on="player_id",
        how="left",
    )
    merged["_unmatched"] = merged["position"].isna()
    merged.loc[merged["_unmatched"], "att_z"] = p20_att
    merged.loc[merged["_unmatched"], "def_z"] = p20_def

    rows: list[dict] = []
    for team, grp in merged.groupby("team"):
        total_em = float(grp["exp_minutes"].sum())
        matched_em = float(grp.loc[~grp["_unmatched"], "exp_minutes"].sum())
        coverage = matched_em / total_em if total_em > 0 else 0.0

        atk_mask = grp["position"].isin(_ATTACK_POSITIONS) | grp["_unmatched"]
        attack = _weighted_mean(grp.loc[atk_mask, "att_z"], grp.loc[atk_mask, "exp_minutes"])

        def_mask = grp["position"].isin(_DEFENSE_POSITIONS) | grp["_unmatched"]
        defense = _weighted_mean(grp.loc[def_mask, "def_z"], grp.loc[def_mask, "exp_minutes"])

        gk_grp = grp[grp["position"] == "Goalkeeper"]
        if gk_grp.empty or gk_grp["gk_z"].isna().all():
            sr_z = gk_grp["player_id"].map(save_rate_z_map)
            gk = _weighted_mean(sr_z, gk_grp["exp_minutes"])
        else:
            gk = _weighted_mean(gk_grp["gk_z"], gk_grp["exp_minutes"])

        rows.append(
            {
                "team": team,
                "attack": attack,
                "defense": defense,
                "gk": gk,
                "coverage": coverage,
                "n_players": len(grp),
            }
        )

    return pd.DataFrame(rows, columns=["team", "attack", "defense", "gk", "coverage", "n_players"])


def style_profile(
    lineups_by_fixture: dict[int, list[dict]],
    team_name: str,
) -> dict:
    """Formation-usage distribution and mean line composition for a national team.

    Returns a dict::

        {
            "formations": {"4-3-3": 0.6, "4-4-2": 0.4, ...},
            "lines": {"G": float, "D": float, "M": float, "F": float},
        }

    Frequencies in ``formations`` sum to 1.0.  ``lines`` values are the mean
    count of startXI players per positional code across all team appearances.
    """
    formation_counts: dict[str, int] = {}
    line_totals: dict[str, float] = {"G": 0.0, "D": 0.0, "M": 0.0, "F": 0.0}
    n_appearances = 0

    for blocks in lineups_by_fixture.values():
        for block in blocks:
            if (block.get("team") or {}).get("name") != team_name:
                continue
            formation = block.get("formation", "")
            if formation:
                formation_counts[formation] = formation_counts.get(formation, 0) + 1
            n_appearances += 1
            for entry in block.get("startXI", []):
                pos = (entry.get("player") or {}).get("pos", "")
                if pos in line_totals:
                    line_totals[pos] += 1.0

    if n_appearances == 0:
        return {"formations": {}, "lines": {"G": 0.0, "D": 0.0, "M": 0.0, "F": 0.0}}

    total_f = sum(formation_counts.values())
    formations = {f: count / total_f for f, count in formation_counts.items()}
    lines = {k: v / n_appearances for k, v in line_totals.items()}
    return {"formations": formations, "lines": lines}


def assemble_expected_minutes(
    client,
    intl_comp_seasons: list[tuple[int, int]],
) -> pd.DataFrame:
    """Fetch international lineups and return per-player expected minutes with team.

    Returns columns: ``player_id, team, exp_minutes``.
    """
    lineups_by_fixture: dict[int, list[dict]] = {}
    fixture_dates: dict[int, str] = {}

    for comp_id, season in intl_comp_seasons:
        fixtures = fetch_fixtures(client, comp_id, season)
        for f in fixtures:
            fix = f.get("fixture") or {}
            fid = fix.get("id")
            date = (fix.get("date") or "")[:10]
            if fid and date:
                fixture_dates[fid] = date
        for fid in fixture_ids(fixtures):
            lineups = fetch_lineups(client, fid)
            if lineups:
                lineups_by_fixture[fid] = lineups

    appearances = national_team_appearances(lineups_by_fixture, fixture_dates)
    exp_min = _expected_minutes(appearances)

    team_map = appearances.set_index("player_id")["team"].to_dict()
    exp_min = exp_min.copy()
    exp_min["team"] = exp_min["player_id"].map(team_map)
    return exp_min[["player_id", "team", "exp_minutes"]].dropna(subset=["team"])


def build_clubform(
    client,
    club_league_seasons: list[tuple[int, int]],
    intl_comp_seasons: list[tuple[int, int]],
    strength_index: dict[int, float],
) -> pd.DataFrame:
    """Top-level orchestrator: club data → player quality → squad composites.

    Reads exclusively from cache via the API-Football client. Returns a
    per-team DataFrame with columns ``attack, defense, gk, coverage, n_players``.

    Parameters
    ----------
    client:
        :class:`~cupcast.v2.fetch.client.ApiFootballClient` instance.
    club_league_seasons:
        Each tuple is ``(league_id, season_year)`` for club player stats.
    intl_comp_seasons:
        Each tuple is ``(competition_id, season_year)`` for international lineups.
    strength_index:
        Output of :func:`~cupcast.v2.clubform.league_strength.league_strength_index`.
    """
    player_responses_by_league: dict[int, list[dict]] = {}
    for league_id, season in club_league_seasons:
        responses = fetch_players(client, league_id, season)
        player_responses_by_league.setdefault(league_id, []).extend(responses)

    player_quality = assemble_player_quality(player_responses_by_league, strength_index)
    exp_min = assemble_expected_minutes(client, intl_comp_seasons)
    return squad_profiles(player_quality, exp_min)
