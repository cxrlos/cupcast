from __future__ import annotations

import argparse

from dotenv import load_dotenv

from cupcast.fetch import catalog
from cupcast.fetch.api_football import ApiFootballClient, ApiFootballError, QuotaExceededError

# Training window for the Dixon-Coles fit and the Elo run-up.
SEASON_FIRST = 2015
SEASON_LAST = 2026

# League names as published by API-Football. `plan` reports any that fail to
# resolve against the live catalog so this list can be corrected.
INTERNATIONAL_LEAGUES = [
    "World Cup",
    "World Cup - Qualification Europe",
    "World Cup - Qualification South America",
    "World Cup - Qualification Africa",
    "World Cup - Qualification Asia",
    "World Cup - Qualification CONCACAF",
    "World Cup - Qualification Oceania",
    "World Cup - Qualification Intercontinental Play-offs",
    "Friendlies",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "Euro Championship",
    "Copa America",
    "Africa Cup of Nations",
    "Asian Cup",
    "CONCACAF Gold Cup",
]


def build_plan(
    client: ApiFootballClient,
    season_first: int = SEASON_FIRST,
    season_last: int = SEASON_LAST,
) -> tuple[list[dict], list[str]]:
    leagues = catalog.load_leagues(client)
    plan: list[dict] = []
    missing: list[str] = []
    for name in INTERNATIONAL_LEAGUES:
        entry = catalog.find_league(leagues, name)
        if entry is None:
            missing.append(name)
            continue
        for year in catalog.seasons_between(entry, season_first, season_last):
            plan.append(
                {"league": entry["league"]["id"], "name": entry["league"]["name"], "season": year}
            )
    return plan, missing


def fixtures_params(item: dict) -> dict:
    return {"league": item["league"], "season": item["season"]}


def cmd_plan(client: ApiFootballClient, season_first: int, season_last: int) -> None:
    plan, missing = build_plan(client, season_first, season_last)
    pending = [item for item in plan if not client.is_cached("fixtures", fixtures_params(item))]
    for item in plan:
        state = "cached" if item not in pending else "pending"
        print(f"{state:8} {item['name']} {item['season']} (league {item['league']})")
    print(f"\n{len(plan) - len(pending)} cached, {len(pending)} pending")
    print(f"requests used today: {client.requests_today()}/{client.daily_budget}")
    for name in missing:
        print(f"UNRESOLVED league name: {name!r} — fix INTERNATIONAL_LEAGUES via `search`")


def cmd_run(client: ApiFootballClient, season_first: int, season_last: int) -> None:
    plan, missing = build_plan(client, season_first, season_last)
    for name in missing:
        print(f"skipping unresolved league name: {name!r}")
    fetched, rejected = 0, 0
    for item in plan:
        params = fixtures_params(item)
        if client.is_cached("fixtures", params):
            continue
        try:
            matches = client.get_response("fixtures", params)
        except QuotaExceededError as exc:
            print(f"stopping: {exc}")
            break
        except ApiFootballError as exc:
            rejected += 1
            print(f"rejected {item['name']} {item['season']}: {exc}")
            continue
        fetched += 1
        print(f"fetched  {item['name']} {item['season']}: {len(matches)} fixtures")
    print(f"\nleague-seasons fetched this run: {fetched} (rejected by plan: {rejected})")
    print(f"requests used today: {client.requests_today()}/{client.daily_budget}")


def cmd_search(client: ApiFootballClient, term: str) -> None:
    for entry in catalog.search_leagues(catalog.load_leagues(client), term):
        league, country = entry["league"], (entry.get("country") or {}).get("name", "?")
        years = catalog.seasons_between(entry, SEASON_FIRST, SEASON_LAST)
        print(f"{league['id']:6} {league['name']} [{country}] seasons {years}")


def cmd_status(client: ApiFootballClient) -> None:
    fixtures_dir = client.cache_dir / "fixtures"
    cached = len(list(fixtures_dir.glob("*.json"))) if fixtures_dir.exists() else 0
    print(f"cached fixture pages: {cached}")
    print(f"requests used today: {client.requests_today()}/{client.daily_budget}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="cupcast-pull", description="Rationed API-Football pulls")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="resolve leagues and show cached/pending league-seasons")
    run = sub.add_parser("run", help="fetch pending league-seasons until the daily budget runs out")
    for cmd in (plan, run):
        cmd.add_argument("--from", dest="season_first", type=int, default=SEASON_FIRST)
        cmd.add_argument("--to", dest="season_last", type=int, default=SEASON_LAST)
    search = sub.add_parser("search", help="search the cached league catalog by name")
    search.add_argument("term")
    sub.add_parser("status", help="show cache and ledger usage")

    args = parser.parse_args()
    client = ApiFootballClient()
    if args.command == "plan":
        cmd_plan(client, args.season_first, args.season_last)
    elif args.command == "run":
        cmd_run(client, args.season_first, args.season_last)
    elif args.command == "search":
        cmd_search(client, args.term)
    elif args.command == "status":
        cmd_status(client)


if __name__ == "__main__":
    main()
