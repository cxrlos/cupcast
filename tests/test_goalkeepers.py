import numpy as np
import pandas as pd
import pytest

from cupcast.v1.features.goalkeepers import keeper_zscores
from cupcast.v1.sim.knockout import shootout_probability


def test_shootout_probability_symmetric_and_capped():
    assert shootout_probability(0.0) == 0.5
    assert shootout_probability(1.0) == pytest.approx(0.52)
    assert shootout_probability(1.0) + shootout_probability(-1.0) == pytest.approx(1.0)
    assert shootout_probability(10.0) == pytest.approx(0.55)
    assert shootout_probability(-10.0) == pytest.approx(0.45)


def test_keeper_zscores_from_synthetic_files(tmp_path):
    squads = pd.DataFrame(
        {
            "team": ["Alphaland"] * 3 + ["Betatia"] * 3,
            "position": ["GK", "GK", "DF"] * 2,
            "name": ["Ace Keeper", "Backup One", "Some Defender"]
            + ["Bad Keeper", "Backup Two", "Other Defender"],
            "caps": [50, 10, 40, 45, 8, 30],
        }
    )
    squads_path = tmp_path / "squads.csv"
    squads.to_csv(squads_path, index=False)
    keepers = pd.DataFrame(
        {
            "player": ["Ace Keeper", "Bad Keeper", "Mid Keeper A", "Mid Keeper B"],
            "expected_psxg+/-": [9.0, -9.0, 0.5, -0.5],
            "playing time_90s": [30.0, 30.0, 30.0, 30.0],
        }
    )
    keeper_path = tmp_path / "keeper_adv.csv"
    keepers.to_csv(keeper_path, index=False)

    zscores = keeper_zscores(squads_path=squads_path, keeper_path=keeper_path)
    assert set(zscores) == {"Alphaland", "Betatia"}
    assert zscores["Alphaland"] > 0 > zscores["Betatia"]
    assert np.isclose(zscores["Alphaland"], -zscores["Betatia"])


def test_keeper_zscores_empty_without_data(tmp_path):
    assert keeper_zscores(keeper_path=tmp_path / "missing.csv") == {}
