from cupcast.v2.fetch import endpoints


class StubClient:
    def __init__(self, by_endpoint):
        self.by_endpoint = by_endpoint
        self.seen = []

    def get_response(self, endpoint, params=None):
        self.seen.append((endpoint, dict(params or {})))
        return self.by_endpoint[endpoint]


def test_fetch_fixtures_passes_league_season():
    client = StubClient({"fixtures": [{"fixture": {"id": 7}}]})
    out = endpoints.fetch_fixtures(client, league=1, season=2022)
    assert out == [{"fixture": {"id": 7}}]
    assert client.seen == [("fixtures", {"league": 1, "season": 2022})]


def test_fetch_players_uses_players_endpoint():
    client = StubClient({"players": [{"player": {"id": 3}}]})
    endpoints.fetch_players(client, league=39, season=2025)
    assert client.seen == [("players", {"league": 39, "season": 2025})]


def test_fixture_ids_keeps_only_finished():
    fixtures = [
        {"fixture": {"id": 1, "status": {"short": "FT"}}},
        {"fixture": {"id": 2, "status": {"short": "NS"}}},
        {"fixture": {"id": 3, "status": {"short": "AET"}}},
        {"fixture": {"id": 5, "status": {"short": "PEN"}}},
        {"fixture": {"status": {"short": "FT"}}},  # final but no id -> skipped safely
    ]
    assert endpoints.fixture_ids(fixtures) == [1, 3, 5]
