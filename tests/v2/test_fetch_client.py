import json

import pytest

from cupcast.v2.fetch.client import (
    ApiFootballClient,
    QuotaExceededError,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, pages):
        # pages: list of payloads returned in order
        self._pages = list(pages)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return FakeResponse(self._pages.pop(0))


def _client(tmp_path, session, **kw):
    return ApiFootballClient(
        cache_dir=tmp_path / "api_football",
        session=session,
        api_key="test-key",
        min_interval=0.0,
        **kw,
    )


def test_get_caches_first_fetch(tmp_path):
    payload = {"response": [{"x": 1}], "paging": {"current": 1, "total": 1}}
    session = FakeSession([payload])
    client = _client(tmp_path, session)
    first = client.get("fixtures", {"league": 1, "season": 2022})
    second = client.get("fixtures", {"league": 1, "season": 2022})
    assert first == second == payload
    assert len(session.calls) == 1  # second read came from cache
    assert client.is_cached("fixtures", {"league": 1, "season": 2022})


def test_get_response_follows_paging(tmp_path):
    p1 = {"response": [{"id": 1}], "paging": {"current": 1, "total": 2}}
    p2 = {"response": [{"id": 2}], "paging": {"current": 2, "total": 2}}
    session = FakeSession([p1, p2])
    client = _client(tmp_path, session)
    items = client.get_response("players", {"league": 39, "season": 2025})
    assert [i["id"] for i in items] == [1, 2]
    assert session.calls[0]["params"].get("page") is None
    assert session.calls[1]["params"]["page"] == 2


def test_ledger_counts_and_budget(tmp_path):
    payload = {"response": [], "paging": {"current": 1, "total": 1}}
    session = FakeSession([payload])
    client = _client(tmp_path, session, daily_budget=1)
    client.get("fixtures", {"league": 1, "season": 2018})
    assert client.requests_today() == 1
    assert json.loads(client.ledger_path.read_text())  # ledger written
    with pytest.raises(QuotaExceededError):
        client.get("fixtures", {"league": 2, "season": 2018})  # over budget
