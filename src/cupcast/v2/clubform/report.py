"""Coverage report for the cupcast.v2.clubform subsystem.

Run as ``python -m cupcast.v2.clubform.report`` after a full data pull to
inspect per-team composite quality and squad-minutes coverage.
"""

from __future__ import annotations

import pandas as pd


def coverage_report(
    profiles: pd.DataFrame,
    wc_teams: list[str] | None = None,
) -> pd.DataFrame:
    """Return a per-team coverage summary sorted by coverage descending.

    Parameters
    ----------
    profiles:
        Output of :func:`~cupcast.v2.clubform.composite.build_clubform` —
        columns ``team, attack, defense, gk, coverage, n_players``.
    wc_teams:
        When provided, only rows whose ``team`` is in this list are returned.
    """
    cols = ["team", "attack", "defense", "gk", "coverage", "n_players"]
    out = profiles[cols].copy()
    if wc_teams is not None:
        out = out[out["team"].isin(wc_teams)]
    return out.sort_values("coverage", ascending=False).reset_index(drop=True)


def _build_club_to_league(
    player_responses_by_league: dict[int, list[dict]],
) -> dict[str, int]:
    """Extract normalized club name → league id from API-Football player responses."""
    from cupcast.v2.clubform.league_strength import normalize_club_name

    mapping: dict[str, int] = {}
    for league_id, responses in player_responses_by_league.items():
        for entry in responses:
            stats = entry.get("statistics") or []
            if stats:
                team_name = (stats[0].get("team") or {}).get("name")
                if team_name:
                    mapping[normalize_club_name(team_name)] = league_id
    return mapping


def main() -> None:
    """Run the clubform pipeline against the real cache and print coverage."""
    from dotenv import load_dotenv

    from cupcast.v2.clubform.composite import build_clubform
    from cupcast.v2.clubform.league_strength import (
        fetch_clubelo,
        league_strength_index,
    )
    from cupcast.v2.fetch import catalog as catalog_mod
    from cupcast.v2.fetch.client import ApiFootballClient
    from cupcast.v2.fetch.endpoints import fetch_players

    load_dotenv()

    client = ApiFootballClient()
    cat = client.get_response("leagues")

    import pandas as _pd

    squads = _pd.read_csv("data/processed/squads.csv")

    # Resolve club leagues and international competitions the same way as pull.py.
    club_entries = catalog_mod.club_leagues_from_squads(cat, squads)
    intl_entries = catalog_mod.resolve_competitions(cat, catalog_mod.INTERNATIONAL_COMPETITIONS)

    club_league_seasons: list[tuple[int, int]] = [
        (e["league"]["id"], season)
        for e in club_entries
        for season in (2024, 2025)
    ]
    intl_comp_seasons: list[tuple[int, int]] = [
        (e["league"]["id"], season)
        for e in intl_entries
        for season in catalog_mod.seasons_between(e, 2018, 2026)
    ]

    # Build club_to_league for the league-strength index.
    player_responses_by_league: dict[int, list[dict]] = {}
    for league_id, season in club_league_seasons:
        responses = fetch_players(client, league_id, season)
        player_responses_by_league.setdefault(league_id, []).extend(responses)

    club_to_league = _build_club_to_league(player_responses_by_league)

    clubelo = fetch_clubelo()
    strength = league_strength_index(clubelo, club_to_league)

    profiles = build_clubform(client, club_league_seasons, intl_comp_seasons, strength)

    report = coverage_report(profiles)
    mean_cov = report["coverage"].mean()
    median_cov = report["coverage"].median()
    print(f"teams={len(report)}  mean_coverage={mean_cov:.2%}  median_coverage={median_cov:.2%}\n")
    print(report.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
