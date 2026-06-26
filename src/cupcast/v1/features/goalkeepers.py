from __future__ import annotations

from pathlib import Path

import pandas as pd

from cupcast.v1.features.expected_minutes import expected_minutes_per_match
from cupcast.v1.features.squad_strength import SQUADS, _find_column, normalize_name

KEEPER_ADV = Path("data/processed/fbref_keeper_adv_2025_26.csv")
MIN_NINETIES = 5.0


def keeper_zscores(
    squads_path: Path = SQUADS, keeper_path: Path = KEEPER_ADV
) -> dict[str, float]:
    """First-choice keeper shot-stopping (PSxG+/- per 90) as a z-score per team.

    Empty dict when keeper data is absent; consumers treat missing teams as
    average (z = 0).
    """
    if not keeper_path.exists() or not squads_path.exists():
        return {}
    raw = pd.read_csv(keeper_path)
    player_col = _find_column(raw, "player")
    psxg_col = next(
        (c for c in raw.columns if "psxg+/-" in c.lower() and "/90" not in c.lower()),
        None,
    )
    nineties_col = _find_column(raw, "90s")
    if player_col is None or psxg_col is None or nineties_col is None:
        raise OSError(
            f"unrecognized keeper_adv columns in {keeper_path}: {list(raw.columns)[:10]}..."
        )
    keepers = raw[[player_col, psxg_col, nineties_col]].copy()
    keepers.columns = ["player", "psxg_diff", "nineties"]
    keepers["psxg_diff"] = pd.to_numeric(keepers["psxg_diff"], errors="coerce")
    keepers["nineties"] = pd.to_numeric(keepers["nineties"], errors="coerce")
    keepers = keepers.dropna().query("nineties >= @MIN_NINETIES")
    keepers["per90"] = keepers["psxg_diff"] / keepers["nineties"]
    keepers["z"] = (keepers["per90"] - keepers["per90"].mean()) / keepers["per90"].std()
    keepers["key"] = keepers["player"].map(normalize_name)
    keepers = keepers.sort_values("nineties", ascending=False).drop_duplicates("key")

    squads = expected_minutes_per_match(pd.read_csv(squads_path))
    starters = squads[(squads["position"] == "GK") & (squads["expected_minutes"] > 0)].copy()
    starters["key"] = starters["name"].map(normalize_name)
    merged = starters.merge(keepers[["key", "z"]], on="key", how="inner")
    return dict(zip(merged["team"], merged["z"], strict=False))
