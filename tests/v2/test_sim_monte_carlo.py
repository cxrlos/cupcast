"""TDD tests for cupcast.v2.sim.monte_carlo."""

from __future__ import annotations

import numpy as np
import pytest

from cupcast.v2.sim.monte_carlo import TournamentDetails, run_tournament, simulate_tournament

_PROB_COLUMNS = [
    "p_group_win",
    "p_group_runner_up",
    "p_group_third",
    "p_qualify",
    "p_r16",
    "p_qf",
    "p_sf",
    "p_final",
    "p_champion",
    "p_runner_up",
    "p_podium_third",
]


class StubPosterior:
    """Duck-typed stub: rate() returns slightly home-biased Poisson rates.

    Accepts and ignores extra kwargs so the same instance works across all
    internal callers (predict.score_matrix forwards host_home/host_away to rate).
    """

    rho = 0.0

    def rate(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
    ) -> tuple[float, float]:
        lam = 1.6 + 0.2 * host_home
        nu = 1.2 + 0.2 * host_away
        return lam, nu


_STUB = StubPosterior()
_N_SIMS = 200
_SEED = 1


@pytest.fixture(scope="module")
def result() -> TournamentDetails:
    return simulate_tournament(_STUB, n_sims=_N_SIMS, seed=_SEED)


def test_table_has_48_rows(result: TournamentDetails) -> None:
    assert len(result.table) == 48


def test_table_has_all_probability_columns(result: TournamentDetails) -> None:
    for col in _PROB_COLUMNS:
        assert col in result.table.columns, f"Missing column: {col}"


def test_all_probability_columns_in_unit_interval(result: TournamentDetails) -> None:
    for col in _PROB_COLUMNS:
        vals = result.table[col]
        assert vals.min() >= -1e-9, f"{col} has value below 0"
        assert vals.max() <= 1.0 + 1e-9, f"{col} has value above 1"


def test_p_champion_sums_to_one(result: TournamentDetails) -> None:
    total = result.table["p_champion"].sum()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_p_qualify_geq_p_champion_per_team(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_qualify"] < df["p_champion"] - 1e-9]
    assert bad.empty, f"p_qualify < p_champion for: {bad['team'].tolist()}"


def test_p_qualify_geq_p_r16_per_team(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_qualify"] < df["p_r16"] - 1e-9]
    assert bad.empty, f"p_qualify < p_r16 for: {bad['team'].tolist()}"


def test_p_r16_geq_p_qf(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_r16"] < df["p_qf"] - 1e-9]
    assert bad.empty


def test_p_qf_geq_p_sf(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_qf"] < df["p_sf"] - 1e-9]
    assert bad.empty


def test_p_sf_geq_p_final(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_sf"] < df["p_final"] - 1e-9]
    assert bad.empty


def test_p_final_geq_p_champion(result: TournamentDetails) -> None:
    df = result.table
    bad = df[df["p_final"] < df["p_champion"] - 1e-9]
    assert bad.empty


def test_table_sorted_by_p_champion_descending(result: TournamentDetails) -> None:
    champ = result.table["p_champion"].to_numpy()
    assert (np.diff(champ) <= 1e-9).all(), "Table must be sorted descending by p_champion"


def test_exact_reproducibility() -> None:
    a = simulate_tournament(_STUB, n_sims=_N_SIMS, seed=_SEED)
    b = simulate_tournament(_STUB, n_sims=_N_SIMS, seed=_SEED)
    np.testing.assert_array_equal(
        a.table["p_champion"].to_numpy(),
        b.table["p_champion"].to_numpy(),
    )


def test_run_tournament_returns_dataframe(result: TournamentDetails) -> None:
    import pandas as pd

    df = run_tournament(_STUB, n_sims=_N_SIMS, seed=_SEED)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 48


def test_tournament_details_fields(result: TournamentDetails) -> None:
    assert result.winners.shape == (12, _N_SIMS)
    assert result.runners.shape == (12, _N_SIMS)
    assert result.thirds.shape == (12, _N_SIMS)
    assert result.qualifies.shape == (12, _N_SIMS)
    assert isinstance(result.match_winner, dict)
    assert isinstance(result.match_loser, dict)
