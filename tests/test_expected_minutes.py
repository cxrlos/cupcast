import numpy as np
import pandas as pd

from cupcast.v1.features.expected_minutes import MATCH_MINUTES, expected_minutes_per_match


def squad_frame():
    rows = []
    for team in ["Alphaland", "Betatia"]:
        for i in range(3):
            rows.append({"team": team, "position": "GK", "name": f"{team} gk{i}", "caps": 30 - i})
        for i in range(9):
            rows.append({"team": team, "position": "DF", "name": f"{team} df{i}", "caps": 60 - i})
        for i in range(8):
            rows.append({"team": team, "position": "MF", "name": f"{team} mf{i}", "caps": 50 - i})
        for i in range(6):
            rows.append({"team": team, "position": "FW", "name": f"{team} fw{i}", "caps": 40 - i})
    return pd.DataFrame(rows)


def test_team_minutes_sum_to_eleven_slots():
    result = expected_minutes_per_match(squad_frame())
    totals = result.groupby("team")["expected_minutes"].sum()
    assert np.allclose(totals, MATCH_MINUTES * 11)


def test_first_choice_keeper_plays_full_match():
    result = expected_minutes_per_match(squad_frame())
    for _, keepers in result[result["position"] == "GK"].groupby("team"):
        ranked = keepers.sort_values("caps", ascending=False)
        assert ranked.iloc[0]["expected_minutes"] == MATCH_MINUTES
        assert (ranked.iloc[1:]["expected_minutes"] == 0).all()


def test_more_caps_never_means_fewer_minutes_within_position():
    result = expected_minutes_per_match(squad_frame())
    for (_, _), group in result.groupby(["team", "position"]):
        ordered = group.sort_values("caps", ascending=False)
        minutes = ordered["expected_minutes"].to_numpy()
        assert (minutes[:-1] >= minutes[1:] - 1e-9).all()


def test_minutes_weights_normalize():
    result = expected_minutes_per_match(squad_frame())
    assert result.groupby("team")["minutes_weight"].sum().round(9).eq(1.0).all()
