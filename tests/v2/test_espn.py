"""Tests for the cached ESPN client — no network calls."""

from __future__ import annotations

import json

import pytest

from cupcast.v2.fetch.espn import EspnClient, canon


# ---------------------------------------------------------------------------
# Helpers: minimal ESPN JSON shapes
# ---------------------------------------------------------------------------

def _scoreboard(events: list[dict]) -> dict:
    return {"events": events}


def _event(home_name: str, away_name: str, home_score: str, away_score: str, completed: bool) -> dict:
    return {
        "date": "2026-06-12T18:00Z",
        "status": {"type": {"completed": completed}},
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": home_name},
                        "score": home_score,
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": away_name},
                        "score": away_score,
                    },
                ],
                "venue": {"address": {"city": "Los Angeles"}},
            }
        ],
    }


def _standings_payload(groups: list[dict]) -> dict:
    return {"children": groups}


def _group_child(letter: str, entries: list[dict]) -> dict:
    return {
        "name": f"Group {letter}",
        "standings": {"entries": entries},
    }


def _entry(display_name: str, rank: int, points: int, note: str = "") -> dict:
    return {
        "team": {"displayName": display_name},
        "stats": [
            {"name": "rank", "value": rank},
            {"name": "points", "value": points},
        ],
        "note": {"description": note},
    }


# ---------------------------------------------------------------------------
# canon() normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("United States", "usa"),
        ("USA", "usa"),
        ("Cape Verde", "capeverdeislands"),
        ("Cape Verde Islands", "capeverdeislands"),
        ("Côte d'Ivoire", "ivorycoast"),
        ("Ivory Coast", "ivorycoast"),
        ("DR Congo", "congodr"),
        ("Congo DR", "congodr"),
        ("Türkiye", "turkiye"),
        ("Turkey", "turkiye"),
        ("Bosnia & Herzegovina", "bosniaherzegovina"),
        ("Bosnia-Herzegovina", "bosniaherzegovina"),
    ],
)
def test_canon_normalises(raw, expected):
    assert canon(raw) == expected


# ---------------------------------------------------------------------------
# completed_group_results — parses completed events from cache
# ---------------------------------------------------------------------------

def _write_cache(client: EspnClient, url: str, params: dict, payload: dict) -> None:
    import hashlib
    key = json.dumps({"url": url, "params": params}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = client.cache_dir / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_completed_group_results_parses(tmp_path):
    client = EspnClient(cache_dir=tmp_path / "espn")
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    payload = _scoreboard([
        _event("Brazil", "Morocco", "2", "0", completed=True),
        _event("United States", "Türkiye", "1", "1", completed=True),
        _event("France", "Senegal", "3", "1", completed=False),  # not completed → excluded
    ])
    _write_cache(client, scoreboard_url, {"dates": "20260611-20260628"}, payload)

    results = client.completed_group_results("20260611", "20260628")
    assert len(results) == 2
    r0 = results[0]
    assert r0["home"] == "brazil"
    assert r0["away"] == "morocco"
    assert r0["home_goals"] == 2
    assert r0["away_goals"] == 0
    r1 = results[1]
    assert r1["home"] == "usa"
    assert r1["away"] == "turkiye"


def test_completed_group_results_reads_cache_second_call(tmp_path, monkeypatch):
    """Second call must not touch the network."""
    client = EspnClient(cache_dir=tmp_path / "espn")
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    payload = _scoreboard([_event("Brazil", "Morocco", "1", "0", completed=True)])
    _write_cache(client, scoreboard_url, {"dates": "20260611-20260628"}, payload)

    # Monkeypatch _fetch so any network call raises
    def _no_network(self, url, params):  # noqa: ARG001
        raise AssertionError("network was called on a cached request")

    monkeypatch.setattr(EspnClient, "_fetch", _no_network)

    r1 = client.completed_group_results("20260611", "20260628")
    r2 = client.completed_group_results("20260611", "20260628")
    assert r1 == r2


# ---------------------------------------------------------------------------
# scheduled_fixtures — excludes completed, includes venue_city
# ---------------------------------------------------------------------------

def test_scheduled_fixtures_excludes_completed(tmp_path):
    client = EspnClient(cache_dir=tmp_path / "espn")
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    payload = _scoreboard([
        _event("Brazil", "Morocco", "2", "0", completed=True),   # excluded
        _event("France", "Senegal", "0", "0", completed=False),  # included
    ])
    _write_cache(client, scoreboard_url, {"dates": "20260620-20260621"}, payload)

    fixtures = client.scheduled_fixtures("20260620", "20260621")
    assert len(fixtures) == 1
    assert fixtures[0]["home"] == "france"
    assert fixtures[0]["venue_city"] == "Los Angeles"


# ---------------------------------------------------------------------------
# final_standings — parses groups, rank-sorts, applies canon
# ---------------------------------------------------------------------------

def test_final_standings_parses(tmp_path):
    client = EspnClient(cache_dir=tmp_path / "espn")
    standings_url = "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"
    payload = _standings_payload([
        _group_child(
            "A",
            [
                _entry("Mexico", 1, 9, "Advance to Round of 32"),
                _entry("South Korea", 2, 6),
                _entry("South Africa", 3, 3),
                _entry("Czechia", 4, 0),
            ],
        ),
        _group_child(
            "D",
            [
                _entry("United States", 1, 7, "Advance to Round of 32"),
                _entry("Paraguay", 2, 5),
                _entry("Türkiye", 3, 3),
                _entry("Australia", 4, 1),
            ],
        ),
    ])
    _write_cache(client, standings_url, {"season": 2026}, payload)

    standings = client.final_standings(2026)
    assert set(standings.keys()) == {"A", "D"}

    group_a = standings["A"]
    assert group_a[0]["team"] == "mexico"
    assert group_a[0]["rank"] == 1
    assert group_a[0]["points"] == 9
    assert "Round of 32" in group_a[0]["note"]

    group_d = standings["D"]
    assert group_d[0]["team"] == "usa"
    assert group_d[2]["team"] == "turkiye"
