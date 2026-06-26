"""TDD for cupcast.v2.model.wc2026 (Plan 7 Task 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cupcast.v2.model.fit import Posterior

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _fixture(
    home: str,
    away: str,
    hg: int | None,
    ag: int | None,
    status: str = "FT",
    round_: str = "Group Stage - 1",
    date: str = "2026-06-11T18:00:00+00:00",
) -> dict:
    return {
        "fixture": {"id": 1, "date": date, "status": {"short": status}},
        "league": {"round": round_},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "goals": {"home": hg, "away": ag},
    }


class _StubClient:
    def __init__(self, fixtures: list[dict]) -> None:
        self._fixtures = fixtures

    def get_response(self, endpoint: str, params: dict | None = None) -> list[dict]:
        assert endpoint == "fixtures"
        return self._fixtures


def _posterior(teams: tuple[str, ...], gamma: float = 0.3) -> Posterior:
    """Minimal Posterior with equal zero strengths."""
    n = len(teams)
    return Posterior(
        teams=teams,
        attack=np.zeros(n),
        defense=np.zeros(n),
        mu=0.0,
        gamma=gamma,
        rho=-0.1,
    )


def _results(*matches: tuple) -> pd.DataFrame:
    """Build a minimal results DataFrame from (home, away, hg, ag) tuples."""
    rows = []
    for home, away, hg, ag in matches:
        outcome = 0 if hg > ag else (1 if hg == ag else 2)
        rows.append(
            {
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "outcome": outcome,
                "stage": "Group Stage - 1",
            }
        )
    cols = ["home", "away", "home_goals", "away_goals", "outcome", "stage"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# actual_wc2026_results
# ---------------------------------------------------------------------------


class TestActualWc2026Results:
    def _client(self) -> _StubClient:
        return _StubClient(
            [
                _fixture("Mexico", "Poland", 0, 0, date="2026-06-11T18:00:00+00:00"),
                _fixture(
                    "Argentina",
                    "Saudi Arabia",
                    1,
                    2,
                    date="2026-06-12T12:00:00+00:00",
                ),
                _fixture(
                    "France",
                    "Australia",
                    4,
                    1,
                    date="2026-06-13T18:00:00+00:00",
                ),
                _fixture(
                    "Brazil",
                    "Serbia",
                    None,
                    None,
                    status="NS",
                    date="2026-06-14T00:00:00+00:00",
                ),
                _fixture(
                    "Germany",
                    "Japan",
                    2,
                    1,
                    date="2026-06-14T15:00:00+00:00",
                    round_="Group Stage - 2",
                ),
            ]
        )

    def test_excludes_unfinished(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert "Brazil" not in df["home"].values
        assert "Serbia" not in df["away"].values

    def test_correct_row_count(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert len(df) == 4

    def test_outcome_draw(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert df.loc[df["home"] == "Mexico", "outcome"].iloc[0] == 1

    def test_outcome_away_win(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert df.loc[df["home"] == "Argentina", "outcome"].iloc[0] == 2

    def test_outcome_home_win(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert df.loc[df["home"] == "France", "outcome"].iloc[0] == 0

    def test_stage_column(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert "Group Stage - 1" in df["stage"].values
        assert "Group Stage - 2" in df["stage"].values

    def test_sorted_by_date(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert df.iloc[0]["home"] == "Mexico"
        assert df.iloc[-1]["home"] == "Germany"

    def test_null_goals_excluded_despite_final_status(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        client = _StubClient(
            [
                _fixture("A", "B", None, None, status="FT"),
                _fixture("C", "D", 1, 0),
            ]
        )
        df = actual_wc2026_results(client)
        assert len(df) == 1
        assert df.iloc[0]["home"] == "C"

    def test_expected_columns(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(self._client())
        assert list(df.columns) == ["home", "away", "home_goals", "away_goals", "outcome", "stage"]

    def test_empty_fixture_list(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        df = actual_wc2026_results(_StubClient([]))
        assert df.empty
        assert list(df.columns) == ["home", "away", "home_goals", "away_goals", "outcome", "stage"]

    def test_aet_pen_included(self):
        from cupcast.v2.model.wc2026 import actual_wc2026_results

        client = _StubClient(
            [
                _fixture("X", "Y", 1, 1, status="AET"),
                _fixture("A", "B", 1, 1, status="PEN"),
            ]
        )
        df = actual_wc2026_results(client)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# score_predictions
# ---------------------------------------------------------------------------


class TestScorePredictions:
    def test_perfect_gives_zero_log_loss(self):
        from cupcast.v2.model.wc2026 import score_predictions

        y = np.array([0, 1, 2, 0])
        P = np.zeros((4, 3))
        P[np.arange(4), y] = 1.0
        assert score_predictions(P, y)["log_loss"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_gives_zero_rps(self):
        from cupcast.v2.model.wc2026 import score_predictions

        y = np.array([0, 1, 2])
        P = np.zeros((3, 3))
        P[np.arange(3), y] = 1.0
        assert score_predictions(P, y)["rps"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_gives_zero_brier(self):
        from cupcast.v2.model.wc2026 import score_predictions

        y = np.array([0, 1, 2])
        P = np.zeros((3, 3))
        P[np.arange(3), y] = 1.0
        assert score_predictions(P, y)["brier"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_log_loss_equals_ln3(self):
        from cupcast.v2.model.wc2026 import score_predictions

        y = np.array([0, 1, 2, 0, 1])
        P = np.full((5, 3), 1.0 / 3.0)
        assert score_predictions(P, y)["log_loss"] == pytest.approx(np.log(3), rel=1e-6)

    def test_n_matches_count(self):
        from cupcast.v2.model.wc2026 import score_predictions

        y = np.array([0, 1, 2])
        P = np.full((3, 3), 1.0 / 3.0)
        assert score_predictions(P, y)["n"] == 3

    def test_returns_all_keys(self):
        from cupcast.v2.model.wc2026 import score_predictions

        scores = score_predictions(np.array([[0.5, 0.3, 0.2]]), np.array([0]))
        assert set(scores.keys()) == {"n", "log_loss", "brier", "rps"}

    def test_accepts_list_inputs(self):
        from cupcast.v2.model.wc2026 import score_predictions

        P = [[1.0, 0.0, 0.0]]
        y = [0]
        scores = score_predictions(P, y)
        assert scores["log_loss"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# held_out_v2
# ---------------------------------------------------------------------------


class TestHeldOutV2:
    def test_skips_both_unknown_teams(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Argentina", "France", "Brazil"))
        res = _results(
            ("Argentina", "France", 1, 0),  # both known → included
            ("Germany", "Japan", 2, 1),  # both unknown → skipped
        )
        P, y = held_out_v2(posterior, res, hosts=())
        assert len(P) == 1
        assert y[0] == 0

    def test_skips_one_unknown_team(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Argentina", "France"))
        res = _results(
            ("Argentina", "Germany", 1, 0),  # Germany unknown → skipped
            ("Argentina", "France", 2, 1),  # both known → included
        )
        P, y = held_out_v2(posterior, res, hosts=())
        assert len(P) == 1

    def test_p_shape(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Mexico", "USA", "Canada", "Poland"))
        res = _results(("Mexico", "Poland", 0, 0), ("USA", "Canada", 1, 2))
        P, y = held_out_v2(posterior, res, hosts=())
        assert P.shape == (2, 3)
        assert y.shape == (2,)

    def test_probabilities_sum_to_one(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Mexico", "USA", "Canada", "Poland"))
        res = _results(("Mexico", "Poland", 0, 0), ("USA", "Canada", 1, 2))
        P, _ = held_out_v2(posterior, res, hosts=())
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_host_flag_raises_home_win_prob(self):
        """host_home=True adds gamma to the home log-rate, raising win probability."""
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Mexico", "Poland"), gamma=0.5)
        res = _results(("Mexico", "Poland", 1, 0))

        P_with_host, _ = held_out_v2(posterior, res, hosts=("Mexico",))
        P_no_host, _ = held_out_v2(posterior, res, hosts=())

        assert P_with_host[0, 0] > P_no_host[0, 0], (
            "Host advantage should increase Mexico home win probability"
        )

    def test_host_away_flag_raises_away_win_prob(self):
        """host_away=True adds gamma to the away log-rate, raising away win probability."""
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Poland", "Mexico"), gamma=0.5)
        res = _results(("Poland", "Mexico", 1, 2))

        P_with_host, _ = held_out_v2(posterior, res, hosts=("Mexico",))
        P_no_host, _ = held_out_v2(posterior, res, hosts=())

        assert P_with_host[0, 2] > P_no_host[0, 2], (
            "Host advantage (away) should increase Mexico away win probability"
        )

    def test_outcomes_aligned_with_predictions(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Argentina", "France", "Brazil", "Germany"))
        res = _results(
            ("Argentina", "France", 2, 0),  # home win → 0
            ("Brazil", "Germany", 1, 1),  # draw → 1
            ("Germany", "Argentina", 0, 3),  # away win → 2
        )
        P, y = held_out_v2(posterior, res, hosts=())
        assert list(y) == [0, 1, 2]

    def test_empty_results(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("A", "B"))
        empty = pd.DataFrame(
            columns=["home", "away", "home_goals", "away_goals", "outcome", "stage"]
        )
        P, y = held_out_v2(posterior, empty, hosts=())
        assert P.shape == (0, 3)
        assert y.shape == (0,)

    def test_all_teams_unknown(self):
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("A", "B"))
        res = _results(("X", "Y", 1, 0), ("Z", "W", 0, 1))
        P, y = held_out_v2(posterior, res, hosts=())
        assert P.shape == (0, 3)
        assert y.shape == (0,)

    def test_uses_default_hosts(self):
        """Default hosts=(Mexico, USA, Canada); verify Mexico gets host_home treatment."""
        from cupcast.v2.model.wc2026 import held_out_v2

        posterior = _posterior(("Mexico", "Poland"), gamma=0.5)
        res = _results(("Mexico", "Poland", 0, 0))

        P_default, _ = held_out_v2(posterior, res)
        P_no_host, _ = held_out_v2(posterior, res, hosts=())

        # Default should apply host advantage (Mexico in HOSTS) → higher home win prob
        assert P_default[0, 0] > P_no_host[0, 0]
