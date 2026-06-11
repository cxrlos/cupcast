from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from cupcast.fetch.tls import ensure_system_certificates

BASE_URL = "https://v3.football.api-sports.io"
# Free tier allows 100/day; keep headroom for retries and manual checks.
DEFAULT_DAILY_BUDGET = 95
# Free tier also caps at 10 requests/minute.
DEFAULT_MIN_INTERVAL = 6.5


class ApiFootballError(RuntimeError):
    pass


class QuotaExceededError(RuntimeError):
    pass


def _utc_today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


class ApiFootballClient:
    def __init__(
        self,
        cache_dir: str | Path = "data/raw/api_football",
        daily_budget: int = DEFAULT_DAILY_BUDGET,
        session: requests.Session | None = None,
        api_key: str | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.daily_budget = daily_budget
        self.min_interval = min_interval
        self._session = session
        self._api_key = api_key
        self._last_request = 0.0

    @property
    def ledger_path(self) -> Path:
        return self.cache_dir / "ledger.json"

    def requests_today(self) -> int:
        return self._read_ledger().get(_utc_today(), 0)

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        params = params or {}
        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        payload = self._fetch(endpoint, params)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
        return payload

    def get_response(self, endpoint: str, params: dict | None = None) -> list:
        # Only some endpoints accept paging; the first request must omit `page`
        # (e.g. /leagues rejects it) and the payload says how many pages exist.
        params = dict(params or {})
        payload = self.get(endpoint, params)
        items = list(payload.get("response", []))
        total_pages = int((payload.get("paging") or {}).get("total") or 1)
        for page in range(2, total_pages + 1):
            payload = self.get(endpoint, {**params, "page": page})
            items.extend(payload.get("response", []))
        return items

    def is_cached(self, endpoint: str, params: dict | None = None) -> bool:
        return self._cache_path(endpoint, params or {}).exists()

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(canonical.encode()).hexdigest()[:16]
        return self.cache_dir / endpoint.strip("/") / f"{digest}.json"

    def _fetch(self, endpoint: str, params: dict) -> dict:
        used = self.requests_today()
        if used >= self.daily_budget:
            raise QuotaExceededError(
                f"daily request budget reached ({used}/{self.daily_budget}); rerun after the "
                "UTC quota reset or raise the budget"
            )
        for attempt in range(3):
            self._pace()
            self._record_request()
            response = self._http().get(
                f"{BASE_URL}/{endpoint.strip('/')}",
                params=params,
                headers={"x-apisports-key": self._key()},
                timeout=30,
            )
            if response.status_code == 429:
                time.sleep(30.0 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors") or {}
            if errors:
                raise ApiFootballError(f"API-Football rejected {endpoint} {params}: {errors}")
            return payload
        raise ApiFootballError(f"rate-limited three times in a row on {endpoint} {params}")

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _http(self) -> requests.Session:
        if self._session is None:
            ensure_system_certificates()
            self._session = requests.Session()
        return self._session

    def _key(self) -> str:
        if not self._api_key:
            self._api_key = os.environ.get("API_FOOTBALL_KEY", "")
        if not self._api_key:
            raise ApiFootballError(
                "API_FOOTBALL_KEY is not set; copy .env.example to .env and add your key"
            )
        return self._api_key

    def _read_ledger(self) -> dict[str, int]:
        if not self.ledger_path.exists():
            return {}
        return json.loads(self.ledger_path.read_text())

    def _record_request(self) -> None:
        ledger = self._read_ledger()
        today = _utc_today()
        ledger[today] = ledger.get(today, 0) + 1
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
