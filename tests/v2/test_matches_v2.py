from cupcast.v2.features.matches import build_match_table

FIXTURES = {
    "World Cup": [
        {"fixture": {"date": "2022-12-18T15:00:00+00:00", "status": {"short": "PEN"}},
         "league": {"season": 2022},
         "teams": {"home": {"name": "Argentina"}, "away": {"name": "France"}},
         "goals": {"home": 3, "away": 3}},
        {"fixture": {"date": "2026-06-11T18:00:00+00:00", "status": {"short": "NS"}},
         "league": {"season": 2026},
         "teams": {"home": {"name": "Mexico"}, "away": {"name": "Korea"}},
         "goals": {"home": None, "away": None}},
        {"fixture": {"date": "2026-07-19T18:00:00+00:00", "status": {"short": "FT"}},
         "league": {"season": 2026},
         "teams": {"home": {"name": "Brazil"}, "away": {"name": "Spain"}},
         "goals": {"home": 2, "away": 1}},
        {"fixture": {"date": "2022-11-25T18:00:00+00:00", "status": {"short": "FT"}},
         "league": {"season": 2022},
         "teams": {"home": {"name": "Japan"}, "away": {"name": "Germany"}},
         "goals": {"home": 2, "away": None}},  # malformed: away null -> dropped
    ]
}


def test_build_match_table_keeps_finished_with_both_goals():
    table = build_match_table(FIXTURES)
    # PEN (Argentina) + FT (Brazil) kept; NS upcoming and away-null (Japan) dropped
    assert list(table["home"]) == ["Argentina", "Brazil"]
    assert table.iloc[0]["home_goals"] == 3
    assert table.iloc[0]["competition"] == "World Cup"
    assert table.iloc[0]["season"] == 2022
    assert table.iloc[0]["status"] == "PEN"


def test_build_match_table_date_is_utc():
    table = build_match_table(FIXTURES)
    assert "UTC" in str(table["date"].dtype)
