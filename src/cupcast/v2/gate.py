"""Hard precondition for forecasting: API-Football and ESPN must agree.

``verify_sources`` pulls scores, standings, and the Round-of-32 draw from both
sources, reconciles them, and raises ``ReconcileError`` on any disagreement.
The pure ``build_reports`` core takes already-fetched data so it can be tested
without network access.
"""

from __future__ import annotations

from cupcast.v2.fetch import endpoints
from cupcast.v2.reconcile import (
    ReconcileReport,
    assert_sources_agree,
    reconcile_matchups,
    reconcile_scores,
    reconcile_standings,
)
from cupcast.v2.sim.live_bracket import resolve_live_r32
from cupcast.v2.sim.structure import GROUP_LETTERS

_PLACEHOLDER_TOKEN = "winner"
SCORE_WINDOW = ("20260611", "20260628")
R32_WINDOW = ("20260628", "20260704")


def api_group_scores(fixtures: list[dict]) -> list[dict]:
    """Completed group-stage results from API-Football fixtures."""
    out = []
    for f in fixtures:
        if "Group Stage" not in str((f.get("league") or {}).get("round", "")):
            continue
        goals = f.get("goals") or {}
        if goals.get("home") is None:
            continue
        teams = f["teams"]
        out.append(
            {
                "home": teams["home"]["name"],
                "away": teams["away"]["name"],
                "home_goals": goals["home"],
                "away_goals": goals["away"],
            }
        )
    return out


def api_group_standings(standings_response: list[dict]) -> dict[str, list[dict]]:
    """Rank-sorted per-group standings, restricted to the twelve real groups.

    API-Football carries an aggregate ``Group Stage`` row that is not a group;
    filtering to ``GROUP_LETTERS`` drops it.
    """
    groups: dict[str, list[dict]] = {}
    for league in standings_response:
        for grouping in league["league"]["standings"]:
            for row in grouping:
                letter = row["group"].replace("Group ", "").strip()
                if letter not in GROUP_LETTERS:
                    continue
                groups.setdefault(letter, []).append(
                    {"team": row["team"]["name"], "rank": row["rank"]}
                )
    for rows in groups.values():
        rows.sort(key=lambda r: r["rank"])
    return groups


def espn_r32_pairs(espn_fixtures: list[dict]) -> list[frozenset[str]]:
    """ESPN Round-of-32 team pairs, excluding later-round placeholder feeders."""
    pairs = []
    for fixture in espn_fixtures:
        home, away = fixture["home"], fixture["away"]
        if _PLACEHOLDER_TOKEN in home or _PLACEHOLDER_TOKEN in away:
            continue
        pairs.append(frozenset({home, away}))
    return pairs


def build_reports(
    fixtures: list[dict],
    api_standings: list[dict],
    espn_scores: list[dict],
    espn_standings: dict[str, list[dict]],
    espn_fixtures: list[dict],
) -> list[ReconcileReport]:
    """Run the three reconciles from already-fetched source data."""
    r_scores = reconcile_scores(api_group_scores(fixtures), espn_scores)

    api_st = api_group_standings(api_standings)
    r_standings = reconcile_standings(api_st, espn_standings)

    winners = {g: rows[0]["team"] for g, rows in api_st.items()}
    runners = {g: rows[1]["team"] for g, rows in api_st.items()}
    r32_fixtures = [
        f for f in fixtures if str((f.get("league") or {}).get("round", "")) == "Round of 32"
    ]
    our = resolve_live_r32(winners, runners, r32_fixtures)
    r_matchups = reconcile_matchups(our, espn_r32_pairs(espn_fixtures))

    return [r_scores, r_standings, r_matchups]


def verify_sources(
    af_client,
    espn_client,
    season: int = 2026,
    score_window: tuple[str, str] = SCORE_WINDOW,
    r32_window: tuple[str, str] = R32_WINDOW,
) -> list[ReconcileReport]:
    """Fetch from both sources, reconcile, and raise on any disagreement.

    Returns the reconcile reports when the sources agree; raises
    ``ReconcileError`` otherwise. Call this before any forecast run.
    """
    fixtures = endpoints.fetch_fixtures(af_client, 1, season)
    api_standings = af_client.get_response(
        "standings", {"league": 1, "season": season}
    )
    espn_scores = espn_client.completed_group_results(*score_window)
    espn_standings = espn_client.final_standings(season)
    espn_fixtures = espn_client.scheduled_fixtures(*r32_window)

    reports = build_reports(
        fixtures, api_standings, espn_scores, espn_standings, espn_fixtures
    )
    assert_sources_agree(*reports)
    return reports
