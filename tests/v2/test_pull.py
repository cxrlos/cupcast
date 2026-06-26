import pandas as pd

from cupcast.v2.fetch import pull
from cupcast.v2.fetch.client import QuotaExceededError


class FakeClient:
    """Records calls; returns canned league catalog + fixtures; can raise quota."""

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after
        self._catalog = [
            {"league": {"id": 1, "name": "World Cup", "type": "Cup"},
             "country": {"name": "World"}, "seasons": [{"year": 2022}]},
            {"league": {"id": 39, "name": "Premier League", "type": "League"},
             "country": {"name": "England"}, "seasons": [{"year": 2024}, {"year": 2025}]},
        ]

    def _maybe_fail(self):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise QuotaExceededError("budget reached")

    def get_response(self, endpoint, params=None):
        self._maybe_fail()
        self.calls.append((endpoint, dict(params or {})))
        if endpoint == "leagues":
            return self._catalog
        if endpoint == "fixtures":
            return [{"fixture": {"id": 100, "status": {"short": "FT"}}}]
        return [{"ok": True}]

    def is_cached(self, endpoint, params=None):
        return False

    def requests_today(self):
        return len(self.calls)


def _squads(tmp_path):
    p = tmp_path / "squads.csv"
    pd.DataFrame({"club_country": ["ENG"]}).to_csv(p, index=False)
    return p


def test_run_pull_collects_all_layers(tmp_path):
    client = FakeClient()
    plan = pull.PullPlan(intl_first_year=2022, intl_last_year=2022, club_seasons=(2024,))
    summary = pull.run_pull(client, plan, squads_path=_squads(tmp_path))
    assert summary.fixtures_pulled >= 1
    assert summary.lineups_pulled >= 1
    assert not summary.stopped_early
    endpoints_hit = {e for e, _ in client.calls}
    assert {"leagues", "fixtures", "fixtures/lineups", "players", "injuries"} <= endpoints_hit


def test_run_pull_stops_cleanly_on_quota(tmp_path):
    client = FakeClient(fail_after=3)
    plan = pull.PullPlan(intl_first_year=2022, intl_last_year=2022, club_seasons=(2024,))
    summary = pull.run_pull(client, plan, squads_path=_squads(tmp_path))
    assert summary.stopped_early is True
    assert summary.requests_used > 0  # partial count recorded before the stop
