from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from cupcast.v1.features.expected_minutes import expected_minutes_per_match

FBREF_STANDARD = Path("data/processed/fbref_standard_2025_26.csv")
UNDERSTAT_PLAYERS = Path("data/processed/understat_players_2025_26.csv")
SQUADS = Path("data/processed/squads.csv")
# Below this share of expected minutes matched to player data, the squad
# composite is unreliable and the team falls back to the Elo-only prior tier.
COVERAGE_THRESHOLD = 0.35
MIN_PLAYER_MINUTES = 450  # one league season's worth of meaningful sample


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def _find_column(frame: pd.DataFrame, *fragments: str) -> str | None:
    for column in frame.columns:
        lowered = column.lower()
        if all(fragment.lower() in lowered for fragment in fragments):
            return column
    return None


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")
    frame["quality_per90"] = pd.to_numeric(frame["quality_per90"], errors="coerce")
    frame = frame.dropna().query("minutes >= @MIN_PLAYER_MINUTES").copy()
    frame["key"] = frame["player"].map(normalize_name)
    # A duplicate key usually means a mid-season transfer: keep the larger sample.
    frame = frame.sort_values("minutes", ascending=False).drop_duplicates("key")
    frame["quality_z"] = (
        frame["quality_per90"] - frame["quality_per90"].mean()
    ) / frame["quality_per90"].std()
    return frame[["key", "quality_z", "minutes"]]


def load_fbref_standard(path: Path = FBREF_STANDARD) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    player_col = _find_column(raw, "player")
    minutes_col = _find_column(raw, "playing time", "min") or _find_column(raw, "min")
    quality_col = _find_column(raw, "per 90", "npxg+xag") or _find_column(raw, "npxg+xag")
    if player_col is None or minutes_col is None or quality_col is None:
        raise OSError(
            f"unrecognized FBref columns in {path}; expected player/minutes/npxG+xAG, "
            f"got {list(raw.columns)[:12]}..."
        )
    frame = raw[[player_col, minutes_col, quality_col]].copy()
    frame.columns = ["player", "minutes", "quality_per90"]
    return _standardize(frame)


def load_understat_players(path: Path = UNDERSTAT_PLAYERS) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    frame = raw[["player", "minutes", "quality_per90"]].copy()
    return _standardize(frame)


def load_player_quality(
    fbref_path: Path = FBREF_STANDARD, understat_path: Path = UNDERSTAT_PLAYERS
) -> pd.DataFrame | None:
    # FBref (npxG+xAG) preferred; Understat (npxG+xA) is the no-browser
    # fallback with the same league coverage. Both are z-scored within their
    # own pool, so the composite scale is consistent either way.
    players = load_fbref_standard(fbref_path)
    if players is None or players.empty:
        players = load_understat_players(understat_path)
    if players is None or players.empty:
        return None
    return players


def squad_composites(
    squads: pd.DataFrame | None = None,
    squads_path: Path = SQUADS,
    fbref_path: Path = FBREF_STANDARD,
) -> pd.DataFrame:
    """Minutes-weighted squad quality per team, with match coverage.

    Returns one row per team: composite (mean quality_z over expected minutes,
    unmatched players imputed at the matched-pool 20th percentile), coverage
    (share of expected minutes matched to player data), and mean age weighted
    by expected minutes. Composite is NaN when no player data is available.
    """
    if squads is None:
        squads = expected_minutes_per_match(pd.read_csv(squads_path))
    squads = squads.copy()
    squads["key"] = squads["name"].map(normalize_name)
    players = load_player_quality(fbref_path)
    rows = []
    for team, group in squads.groupby("team"):
        weights = group["minutes_weight"].to_numpy()
        ages = group["age"].to_numpy(dtype=float)
        age = float(np.nansum(ages * weights) / weights[~np.isnan(ages)].sum())
        if players is None:
            rows.append({"team": team, "composite": np.nan, "coverage": 0.0, "age": age})
            continue
        merged = group.merge(players, on="key", how="left")
        matched = merged["quality_z"].notna()
        coverage = float(merged.loc[matched, "minutes_weight"].sum())
        floor = players["quality_z"].quantile(0.20)
        quality = merged["quality_z"].fillna(floor).to_numpy()
        composite = float(np.sum(quality * merged["minutes_weight"].to_numpy()))
        rows.append(
            {"team": team, "composite": composite, "coverage": coverage, "age": age}
        )
    return pd.DataFrame(rows)
