"""Cached ESPN public-API client for 2026 World Cup results and standings."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import urllib.request
from pathlib import Path

_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
_STANDINGS = (
    "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"
)
_TIMEOUT = 25

# Maps normalized ESPN token → normalized API-Football token when they differ.
ALIAS: dict[str, str] = {
    "unitedstates": "usa",
    "capeverde": "capeverdeislands",
    "cotedivoire": "ivorycoast",
    "ivorycoast": "ivorycoast",
    "drcongo": "congodr",
    "democraticrepublicofthecongo": "congodr",
    "turkey": "turkiye",
    "bosniaandherzegovina": "bosniaherzegovina",
    "czechrepublic": "czechia",
    "northernireland": "northernireland",
}


def canon(name: str) -> str:
    """Return a normalized token that collapses ESPN and API-Football spellings."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode()
    token = "".join(c for c in ascii_only.lower() if c.isalnum())
    return ALIAS.get(token, token)


class EspnClient:
    """Cache-first client for ESPN's public soccer endpoints."""

    def __init__(self, cache_dir: str | Path = "data/raw/espn") -> None:
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def completed_group_results(
        self,
        start: str = "20260611",
        end: str = "20260628",
    ) -> list[dict]:
        """Return completed group-stage matches with normalized team names."""
        data = self._get(_SCOREBOARD, {"dates": f"{start}-{end}"})
        results = []
        for event in data.get("events", []):
            status = (event.get("status") or {}).get("type") or {}
            if not status.get("completed"):
                continue
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if home is None or away is None:
                continue
            try:
                home_goals = int(home.get("score", 0))
                away_goals = int(away.get("score", 0))
            except (TypeError, ValueError):
                continue
            results.append(
                {
                    "home": canon((home.get("team") or {}).get("displayName", "")),
                    "away": canon((away.get("team") or {}).get("displayName", "")),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )
        return results

    def scheduled_fixtures(self, start: str, end: str) -> list[dict]:
        """Return not-yet-played fixtures with normalized names, date, and venue."""
        data = self._get(_SCOREBOARD, {"dates": f"{start}-{end}"})
        fixtures = []
        for event in data.get("events", []):
            status = (event.get("status") or {}).get("type") or {}
            if status.get("completed"):
                continue
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if home is None or away is None:
                continue
            venue = comp.get("venue") or {}
            city = (venue.get("address") or {}).get("city", "")
            fixtures.append(
                {
                    "home": canon((home.get("team") or {}).get("displayName", "")),
                    "away": canon((away.get("team") or {}).get("displayName", "")),
                    "date": event.get("date", ""),
                    "venue_city": city,
                }
            )
        return fixtures

    def final_standings(self, season: int = 2026) -> dict[str, list[dict]]:
        """Return {group_letter: [{"team","rank","points","note"}]} rank-sorted."""
        data = self._get(_STANDINGS, {"season": season})
        groups: dict[str, list[dict]] = {}
        for child in data.get("children", []):
            group_name: str = child.get("name", "")
            # E.g. "Group A" → "A"
            letter = group_name.split()[-1] if group_name else group_name
            standings_block = child.get("standings") or {}
            entries = standings_block.get("entries") or []
            rows = []
            for entry in entries:
                team_name = (entry.get("team") or {}).get("displayName", "")
                stats = entry.get("stats") or []
                rank = next(
                    (int(s["value"]) for s in stats if s.get("name") == "rank"), 0
                )
                points = next(
                    (int(s["value"]) for s in stats if s.get("name") == "points"), 0
                )
                note = (entry.get("note") or {}).get("description", "")
                rows.append(
                    {
                        "team": canon(team_name),
                        "rank": rank,
                        "points": points,
                        "note": note,
                    }
                )
            rows.sort(key=lambda r: r["rank"])
            groups[letter] = rows
        return groups

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> dict:
        cache_path = self._cache_path(url, params)
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        data = self._fetch(url, params)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
        return data

    def _cache_path(self, url: str, params: dict) -> Path:
        key = json.dumps({"url": url, "params": params}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _fetch(self, url: str, params: dict) -> dict:
        if params:
            qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            full_url = f"{url}?{qs}"
        else:
            full_url = url
        with urllib.request.urlopen(full_url, timeout=_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read())
