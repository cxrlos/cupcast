"""League-strength index derived from median ClubElo per league."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Curated strength for major non-European leagues ClubElo does not rate
# (used as the fallback when a league has <3 ClubElo-matched clubs).
NON_EUROPEAN_LEAGUE_STRENGTH: dict[int, float] = {
    71: 0.90,   # Brazil Serie A
    128: 0.82,  # Liga Profesional Argentina
    262: 0.80,  # Liga MX
    253: 0.78,  # MLS
    307: 0.72,  # Saudi Pro League
}


def normalize_club_name(name: str) -> str:
    """NFKD accent-strip, casefold, and strip a club name for fuzzy matching."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def fetch_clubelo(cache_dir: str | Path = "data/raw/clubelo") -> pd.DataFrame:
    """Return a club-Elo snapshot, reading from cache or fetching via soccerdata.

    Columns: ``team`` (normalized name), ``elo`` (float), ``country``, ``level``.
    The snapshot date defaults to today. The result is persisted under
    *cache_dir* as ``clubelo.csv`` so subsequent calls are free.
    """
    import soccerdata  # deferred — not imported at test-collection time

    cache_path = Path(cache_dir) / "clubelo.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        return frame

    raw: pd.DataFrame = soccerdata.ClubElo().read_by_date()
    # soccerdata returns the frame indexed by team; reset so team is a column.
    frame = raw.reset_index()[["team", "elo", "country", "level"]].copy()
    frame["elo"] = pd.to_numeric(frame["elo"], errors="coerce")
    frame["level"] = pd.to_numeric(frame["level"], errors="coerce")
    frame["team"] = frame["team"].map(normalize_club_name)
    frame = frame.dropna(subset=["team", "elo"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def league_strength_index(
    clubelo: pd.DataFrame,
    club_to_league: dict[str, int],
) -> dict[int, float]:
    """Map each league id to a strength multiplier centered on 1.0.

    Algorithm:
    1. For each league, compute the **median** ClubElo of its matched clubs.
    2. Z-score the per-league medians (population: leagues with ≥3 clubs).
    3. Convert to a multiplier via ``1 + 0.25 * z``, clipped to [0.5, 1.5].
    4. Leagues with fewer than 3 matched clubs get the fallback value 1.0.

    Parameters
    ----------
    clubelo:
        DataFrame with at least columns ``team`` and ``elo``.
    club_to_league:
        Maps normalized club name to a league id (integer).
    """
    elo_by_name = clubelo.set_index("team")["elo"].to_dict()

    league_elos: dict[int, list[float]] = {}
    for club_norm, league_id in club_to_league.items():
        if club_norm in elo_by_name:
            league_elos.setdefault(league_id, []).append(float(elo_by_name[club_norm]))

    # Separate leagues with enough data from sparse ones.
    qualified = {lid: vals for lid, vals in league_elos.items() if len(vals) >= 3}
    sparse = {lid for lid in league_elos if lid not in qualified}

    if not qualified:
        return {
            lid: NON_EUROPEAN_LEAGUE_STRENGTH.get(lid, 1.0)
            for lid in set(club_to_league.values())
        }

    league_ids = list(qualified)
    medians = np.array([np.median(qualified[lid]) for lid in league_ids], dtype=float)

    if len(medians) == 1:
        # Single league — z-score is undefined; multiplier is 1.0 by convention.
        multipliers = np.ones(1)
    else:
        std = medians.std(ddof=1)
        if std == 0.0:
            multipliers = np.ones(len(medians))
        else:
            z = (medians - medians.mean()) / std
            multipliers = np.clip(1.0 + 0.25 * z, 0.5, 1.5)

    result: dict[int, float] = {
        lid: float(m) for lid, m in zip(league_ids, multipliers, strict=True)
    }
    for lid in sparse:
        result[lid] = NON_EUROPEAN_LEAGUE_STRENGTH.get(lid, 1.0)
    # Ensure every league present in the data has an entry: leagues with zero
    # ClubElo matches (e.g. most non-European leagues) never reach league_elos,
    # so set them here to their curated fallback (or a neutral 1.0).
    for lid in set(club_to_league.values()):
        result.setdefault(lid, NON_EUROPEAN_LEAGUE_STRENGTH.get(lid, 1.0))
    return result
