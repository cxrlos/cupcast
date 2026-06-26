from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cupcast.v2.fetch import catalog as catalog_mod
from cupcast.v2.fetch import endpoints
from cupcast.v2.fetch.client import ApiFootballClient, QuotaExceededError


@dataclass
class PullPlan:
    intl_first_year: int = 2018
    intl_last_year: int = 2026
    club_seasons: tuple[int, ...] = (2024, 2025)


@dataclass
class PullSummary:
    fixtures_pulled: int = 0
    lineups_pulled: int = 0
    player_pages: int = 0
    injuries_pulled: int = 0
    requests_used: int = 0
    stopped_early: bool = False
    finished_fixture_ids: list[int] = field(default_factory=list)


def _load_catalog(client) -> list[dict]:
    return client.get_response("leagues")


def run_pull(
    client,
    plan: PullPlan | None = None,
    squads_path: str | Path = "data/processed/squads.csv",
) -> PullSummary:
    plan = plan or PullPlan()
    summary = PullSummary()
    try:
        cat = _load_catalog(client)
        intl = catalog_mod.resolve_competitions(cat, catalog_mod.INTERNATIONAL_COMPETITIONS)
        squads = pd.read_csv(squads_path)
        clubs = catalog_mod.club_leagues_from_squads(cat, squads)

        # (1) international fixtures
        for entry in intl:
            for season in catalog_mod.seasons_between(
                entry, plan.intl_first_year, plan.intl_last_year
            ):
                fixtures = endpoints.fetch_fixtures(client, entry["league"]["id"], season)
                summary.fixtures_pulled += len(fixtures)
                summary.finished_fixture_ids += endpoints.fixture_ids(fixtures)

        # (2) club fixtures + players + injuries
        for entry in clubs:
            lid = entry["league"]["id"]
            for season in plan.club_seasons:
                fixtures = endpoints.fetch_fixtures(client, lid, season)
                summary.fixtures_pulled += len(fixtures)
                summary.finished_fixture_ids += endpoints.fixture_ids(fixtures)
                endpoints.fetch_players(client, lid, season)
                summary.player_pages += 1
                inj = endpoints.fetch_injuries(client, lid, season)
                summary.injuries_pulled += len(inj)

        # (3) lineups for every finished fixture (deduped: one lineup per fixture)
        for fid in dict.fromkeys(summary.finished_fixture_ids):
            if client.is_cached("fixtures/lineups", {"fixture": fid}):
                summary.lineups_pulled += 1
                continue
            endpoints.fetch_lineups(client, fid)
            summary.lineups_pulled += 1
    except QuotaExceededError:
        summary.stopped_early = True

    summary.requests_used = client.requests_today()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="cupcast-v2-data", description="v2 aggressive pull")
    parser.add_argument("--intl-from", type=int, default=2018)
    parser.add_argument("--intl-to", type=int, default=2026)
    args = parser.parse_args()
    load_dotenv()
    client = ApiFootballClient()
    plan = PullPlan(intl_first_year=args.intl_from, intl_last_year=args.intl_to)
    summary = run_pull(client, plan)
    print(
        f"fixtures={summary.fixtures_pulled} lineups={summary.lineups_pulled} "
        f"player_pages={summary.player_pages} injuries={summary.injuries_pulled} "
        f"requests_today={summary.requests_used} stopped_early={summary.stopped_early}"
    )
    if summary.stopped_early:
        print("daily quota reached — rerun tomorrow to resume from cache")


if __name__ == "__main__":
    main()
