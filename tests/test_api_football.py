import json

import pytest

from cupcast.fetch.api_football import (
    ApiFootballClient,
    ApiFootballError,
    QuotaExceededError,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return FakeResponse(self.payloads.pop(0))


def page(items, current=1, total=1):
    return {"response": items, "paging": {"current": current, "total": total}, "errors": {}}


def make_client(tmp_path, payloads, budget=10):
    session = FakeSession(payloads)
    client = ApiFootballClient(
        cache_dir=tmp_path, daily_budget=budget, session=session, api_key="test-key",
        min_interval=0.0,
    )
    return client, session


def test_get_caches_responses(tmp_path):
    client, session = make_client(tmp_path, [page([{"id": 1}])])
    first = client.get("fixtures", {"league": 1, "season": 2026})
    second = client.get("fixtures", {"league": 1, "season": 2026})
    assert first == second
    assert len(session.calls) == 1


def test_quota_enforced_and_ledger_persisted(tmp_path):
    client, _ = make_client(tmp_path, [page([]), page([])], budget=2)
    client.get("fixtures", {"league": 1})
    client.get("fixtures", {"league": 2})
    with pytest.raises(QuotaExceededError):
        client.get("fixtures", {"league": 3})
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert sum(ledger.values()) == 2


def test_cache_hits_do_not_consume_budget(tmp_path):
    client, session = make_client(tmp_path, [page([])], budget=1)
    client.get("fixtures", {"league": 1})
    client.get("fixtures", {"league": 1})
    assert client.requests_today() == 1
    assert len(session.calls) == 1


def test_api_errors_raise_and_are_not_cached(tmp_path):
    bad = {"response": [], "paging": {"current": 1, "total": 1}, "errors": {"token": "invalid"}}
    client, session = make_client(tmp_path, [bad, page([{"id": 1}])])
    with pytest.raises(ApiFootballError):
        client.get("fixtures", {"league": 1})
    assert client.get("fixtures", {"league": 1})["response"] == [{"id": 1}]
    assert len(session.calls) == 2


def test_get_response_walks_pages_without_page_param_on_first_call(tmp_path):
    client, session = make_client(
        tmp_path,
        [page([{"id": 1}], current=1, total=2), page([{"id": 2}], current=2, total=2)],
    )
    items = client.get_response("fixtures", {"league": 1})
    assert items == [{"id": 1}, {"id": 2}]
    assert "page" not in session.calls[0][1]
    assert session.calls[1][1]["page"] == 2


def test_missing_key_raises_without_leaking_anything(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    client = ApiFootballClient(cache_dir=tmp_path, session=FakeSession([page([])]))
    with pytest.raises(ApiFootballError, match="API_FOOTBALL_KEY is not set"):
        client.get("fixtures", {"league": 1})
