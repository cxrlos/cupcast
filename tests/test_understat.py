import pandas as pd
import pytest

from cupcast.v1.features.squad_strength import load_player_quality, load_understat_players
from cupcast.v1.fetch.understat import to_players_table


def soccerdata_shaped_stats():
    frame = pd.DataFrame(
        {
            "league": ["ENG-Premier League"] * 3,
            "season": ["2526"] * 3,
            "team": ["Big Club", "Small Club", "Mid Club"],
            "player": ["Star Forward", "Bench Defender", "Zero Minutes"],
            "minutes": [2700, 300, 0],
            "np_xg": [18.4, 0.4, 0.0],
            "xa": [7.2, 0.1, 0.0],
        }
    )
    return frame.set_index(["league", "season", "team", "player"])


def test_to_players_table_computes_per90_and_drops_zero_minutes():
    table = to_players_table(soccerdata_shaped_stats())
    assert len(table) == 2  # zero-minutes row dropped
    star = table[table["player"] == "Star Forward"].iloc[0]
    assert star["quality_per90"] == pytest.approx((18.4 + 7.2) / 30.0)
    assert set(table.columns) >= {"player", "team", "league", "minutes", "quality_per90"}


def understat_csv(tmp_path):
    frame = pd.DataFrame(
        {
            "player": ["Star Forward", "Mid Player", "Tiny Sample"],
            "minutes": [2700, 1800, 100],
            "quality_per90": [0.85, 0.30, 5.0],
        }
    )
    path = tmp_path / "understat.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_understat_filters_and_zscores(tmp_path):
    players = load_understat_players(understat_csv(tmp_path))
    assert len(players) == 2  # the 100-minute sample is dropped
    assert players["quality_z"].mean() == pytest.approx(0.0)


def test_load_player_quality_falls_back_to_understat(tmp_path):
    players = load_player_quality(
        fbref_path=tmp_path / "missing_fbref.csv",
        understat_path=understat_csv(tmp_path),
    )
    assert players is not None and len(players) == 2
    assert (
        load_player_quality(
            fbref_path=tmp_path / "missing.csv", understat_path=tmp_path / "also_missing.csv"
        )
        is None
    )
