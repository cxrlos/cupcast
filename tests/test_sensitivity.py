import numpy as np
import pandas as pd
import pytest

from cupcast.features.expected_minutes import MATCH_MINUTES, expected_minutes_per_match
from cupcast.model.dixon_coles import DixonColesFit
from cupcast.report.sensitivity import adjust_minutes, conditional_paths, find_player
from cupcast.sim.monte_carlo import simulate_tournament
from cupcast.sim.worldcup2026 import ALL_TEAMS


def squad_frame():
    rows = []
    for team in ["Alphaland", "Betatia"]:
        for position, count, caps0 in (("GK", 3, 30), ("DF", 9, 60), ("MF", 8, 50), ("FW", 6, 40)):
            for i in range(count):
                rows.append(
                    {
                        "team": team,
                        "position": position,
                        "name": f"{team} {position.lower()}{i}",
                        "caps": caps0 - i,
                    }
                )
    return pd.DataFrame(rows)


def adjusted_squads():
    squads = expected_minutes_per_match(squad_frame())
    index = find_player(squads, "Alphaland", "Alphaland mf0")
    return squads, index


def test_find_player_exact_and_missing():
    squads, index = adjusted_squads()
    assert len(index) == 1
    assert find_player(squads, "Alphaland", "Nobody Realname").empty


def test_adjust_minutes_conserves_team_total_within_position():
    squads, index = adjusted_squads()
    before_team = squads[squads["team"] == "Alphaland"]["expected_minutes"].sum()
    before_position = squads[
        (squads["team"] == "Alphaland") & (squads["position"] == "MF")
    ]["expected_minutes"].sum()
    adjusted = adjust_minutes(squads, index, minutes_factor=0.0)
    assert adjusted.loc[index[0], "expected_minutes"] == 0.0
    after_team = adjusted[adjusted["team"] == "Alphaland"]["expected_minutes"].sum()
    after_position = adjusted[
        (adjusted["team"] == "Alphaland") & (adjusted["position"] == "MF")
    ]["expected_minutes"].sum()
    assert after_team == pytest.approx(before_team)
    assert after_team == pytest.approx(MATCH_MINUTES * 11)
    assert after_position == pytest.approx(before_position)
    # other team untouched
    untouched = adjusted[adjusted["team"] == "Betatia"]["expected_minutes"]
    assert untouched.equals(squads[squads["team"] == "Betatia"]["expected_minutes"])


def test_partial_factor_scales_player():
    squads, index = adjusted_squads()
    before = squads.loc[index[0], "expected_minutes"]
    adjusted = adjust_minutes(squads, index, minutes_factor=0.7)
    assert adjusted.loc[index[0], "expected_minutes"] == pytest.approx(before * 0.7)


def wc_fit(seed=5):
    rng = np.random.default_rng(seed)
    n = len(ALL_TEAMS)
    attack = rng.normal(0, 0.4, n)
    defense = rng.normal(0, 0.4, n)
    return DixonColesFit(
        teams=ALL_TEAMS,
        mu=0.1,
        host_advantage=0.25,
        rho=-0.05,
        attack=attack - attack.mean(),
        defense=defense - defense.mean(),
    )


def test_conditional_paths_are_probabilities_consistent_with_marginal():
    details = simulate_tournament(wc_fit(), n_sims=3000, seed=11)
    paths = conditional_paths(details)
    table = details.table.set_index("team")
    for row in paths.itertuples():
        assert row.p_champion == pytest.approx(table.loc[row.team, "p_champion"])
        for value in (row.p_champ_if_group_win, row.p_champ_if_runner_up, row.p_champ_if_third):
            assert np.isnan(value) or 0.0 <= value <= 1.0
        # law of total probability: champions only come from qualified paths
        marginals = table.loc[row.team]
        reconstructed = (
            np.nan_to_num(row.p_champ_if_group_win) * marginals["p_group_win"]
            + np.nan_to_num(row.p_champ_if_runner_up) * marginals["p_group_runner_up"]
            + np.nan_to_num(row.p_champ_if_third)
            * (marginals["p_qualify"] - marginals["p_group_win"] - marginals["p_group_runner_up"])
        )
        assert reconstructed == pytest.approx(row.p_champion, abs=1e-9)
