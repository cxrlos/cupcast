import json

import pandas as pd
import pytest

from cupcast.features.squad_strength import load_player_quality, load_understat_players
from cupcast.fetch.understat import parse_players_payload

PLAYERS = [
    {
        "id": "1",
        "player_name": "Star Forward",
        "time": "2700",
        "npxG": "18.4",
        "xA": "7.2",
        "team_title": "Big Club",
    },
    {
        "id": "2",
        "player_name": "Bench Defender",
        "time": "300",
        "npxG": "0.4",
        "xA": "0.1",
        "team_title": "Small Club",
    },
]


def synthetic_page() -> str:
    blob = json.dumps(PLAYERS)
    escaped = "".join(f"\\x{ord(c):02x}" if c in "[]{}\"'" else c for c in blob)
    return f"<script>var playersData = JSON.parse('{escaped}');</script>"


def test_parse_players_payload_decodes_hex_escapes():
    players = parse_players_payload(synthetic_page())
    assert players[0]["player_name"] == "Star Forward"
    assert players[1]["team_title"] == "Small Club"


def test_parse_players_payload_raises_on_layout_change():
    with pytest.raises(OSError, match="layout changed"):
        parse_players_payload("<html>nothing here</html>")


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
