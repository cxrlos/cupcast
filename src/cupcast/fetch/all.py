from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from cupcast.fetch import fbref, football_data, martj42, odds, squads, understat
from cupcast.fetch.api_football import ApiFootballClient
from cupcast.fetch.pull import cmd_run
from cupcast.fetch.tls import ensure_system_certificates


@dataclass(frozen=True)
class Step:
    name: str
    required: bool
    run: callable
    probe_url: str | None  # None = covered elsewhere or local-only


def _api_football(force: bool) -> None:
    # Always cache-first: a forced refetch would burn the daily quota for
    # byte-identical history, so force is deliberately not applied here.
    cmd_run(ApiFootballClient(), 2022, 2024)


STEPS = (
    Step("api_football", True, _api_football, "https://v3.football.api-sports.io/"),
    Step(
        "martj42",
        True,
        lambda force: martj42.fetch_results_csv(refresh=force) and None,
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    ),
    Step("squads", True, squads.main, "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"),
    Step("understat", True, understat.main, "https://understat.com/league/EPL/2025"),
    Step(
        "football_data",
        False,
        football_data.main,
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    ),
    Step("odds", False, odds.main, "https://api.the-odds-api.com/v4/sports"),
    Step("fbref", False, fbref.main, "https://fbref.com/en/"),
)


def run_all(force: bool) -> int:
    results: list[tuple[str, str]] = []
    for step in STEPS:
        label = f"[{step.name}]"
        print(f"\n=== {label} {'(forced refresh)' if force else ''} ===")
        try:
            step.run(force)
            results.append((step.name, "ok"))
        except Exception as exc:  # noqa: BLE001 — isolate every step
            results.append((step.name, f"FAILED: {exc}"))
            print(f"{label} FAILED: {exc}")
    print("\n=== summary ===")
    failed_required = 0
    for name, status in results:
        required = next(s.required for s in STEPS if s.name == name)
        tag = "required" if required else "optional"
        print(f"{name:15} [{tag}] {status if status == 'ok' else status[:160]}")
        if status != "ok" and required:
            failed_required += 1
    if failed_required:
        print(f"\n{failed_required} required step(s) failed — run --doctor for diagnosis")
    return 1 if failed_required else 0


def doctor() -> int:
    load_dotenv()
    ensure_system_certificates()
    print("=== keys (.env) ===")
    for key in ("API_FOOTBALL_KEY", "ODDS_API_KEY"):
        print(f"{key:18} {'set' if os.environ.get(key) else 'MISSING'}")

    print("\n=== reachability ===")
    for step in STEPS:
        if step.probe_url is None:
            continue
        try:
            response = requests.get(
                step.probe_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) cupcast/0.1"},
                stream=True,
            )
            chain = " -> ".join(str(r.status_code) for r in response.history) or "direct"
            body = next(response.iter_content(80), b"")[:80]
            blocked = b"Blocked" in body or b"security team" in body
            verdict = "BLOCKED (filter page)" if blocked else f"HTTP {response.status_code}"
            print(f"{step.name:15} {verdict:22} redirects: {chain:18} final: {response.url}")
            response.close()
        except Exception as exc:  # noqa: BLE001 — report, don't crash
            print(f"{step.name:15} UNREACHABLE: {type(exc).__name__}: {str(exc)[:90]}")

    print("\n=== cache inventory ===")
    expectations = {
        "data/raw/api_football/fixtures": "22 fixture pages",
        "data/raw/martj42": "results.csv",
        "data/raw/wikipedia": "squads wikitext",
        "data/raw/understat": "5 league JSONs",
        "data/raw/football_data": "55 season CSVs",
        "data/processed": "squads/understat/market CSVs",
    }
    for path, expectation in expectations.items():
        directory = Path(path)
        count = len([f for f in directory.rglob("*") if f.is_file()]) if directory.exists() else 0
        print(f"{path:32} {count:4} files (expect: {expectation})")
    return 0


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        prog="cupcast-fetch-all", description="Run every data fetcher with per-step isolation"
    )
    parser.add_argument(
        "--force", action="store_true", help="refresh existing caches (API-Football excluded)"
    )
    parser.add_argument(
        "--doctor", action="store_true", help="diagnose keys, reachability, and cache state"
    )
    args = parser.parse_args()
    load_dotenv()
    raise SystemExit(doctor() if args.doctor else run_all(args.force))


if __name__ == "__main__":
    main()
