"""TDD for cupcast.compare.wc2026 (Plan 7 Task 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubV1Fit:
    """Minimal DixonColesFit-alike: fixed probs, records calls."""

    def __init__(self, teams: tuple[str, ...], probs=(0.4, 0.3, 0.3)):
        self.teams = teams
        self._probs = probs
        self._calls: list[tuple] = []

    def outcome_probs(
        self,
        home: str,
        away: str,
        host_home: bool = False,
        host_away: bool = False,
    ) -> tuple[float, float, float]:
        self._calls.append((home, away, host_home, host_away))
        return self._probs


def _results(*matches: tuple) -> pd.DataFrame:
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


def _stub_posterior(teams: tuple[str, ...]):
    from cupcast.v2.model.fit import Posterior

    n = len(teams)
    return Posterior(
        teams=teams,
        attack=np.zeros(n),
        defense=np.zeros(n),
        mu=0.0,
        gamma=0.3,
        rho=-0.1,
    )


# ---------------------------------------------------------------------------
# V1_NAME_MAP
# ---------------------------------------------------------------------------


class TestV1NameMap:
    def test_contains_expected_mappings(self):
        from cupcast.compare.wc2026 import V1_NAME_MAP

        assert V1_NAME_MAP["USA"] == "United States"
        assert V1_NAME_MAP["Czechia"] == "Czech Republic"
        assert V1_NAME_MAP["Türkiye"] == "Turkey"
        assert V1_NAME_MAP["Cape Verde Islands"] == "Cape Verde"
        assert V1_NAME_MAP["Congo DR"] == "DR Congo"
        assert V1_NAME_MAP["Bosnia & Herzegovina"] == "Bosnia and Herzegovina"


# ---------------------------------------------------------------------------
# held_out_v1
# ---------------------------------------------------------------------------


class TestHeldOutV1:
    def test_maps_usa_to_united_states(self):
        """'USA' in results → 'United States' looked up in v1 teams."""
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico", "Canada"))
        res = _results(("USA", "Mexico", 1, 0))
        P, y = held_out_v1(v1, res, hosts=())
        assert len(P) == 1
        assert y[0] == 0

    def test_calls_outcome_probs_with_mapped_names(self):
        """outcome_probs receives the v1-mapped name, not the API-Football name."""
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico"))
        res = _results(("USA", "Mexico", 1, 0))
        held_out_v1(v1, res, hosts=())
        assert v1._calls[0][0] == "United States"
        assert v1._calls[0][1] == "Mexico"

    def test_host_home_flag_uses_original_name(self):
        """host_home check uses original API-Football name ('USA'), not mapped."""
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico"))
        res = _results(("USA", "Mexico", 1, 0))
        held_out_v1(v1, res, hosts=("USA",))
        assert v1._calls[0][2] is True

    def test_host_away_flag_uses_original_name(self):
        """host_away check uses original API-Football name."""
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("Mexico", "United States"))
        res = _results(("Mexico", "USA", 0, 1))
        held_out_v1(v1, res, hosts=("USA",))
        assert v1._calls[0][3] is True

    def test_skips_row_with_both_unknown_mapped_teams(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico"))
        res = _results(
            ("USA", "Mexico", 1, 0),       # scorable
            ("Germany", "France", 0, 1),   # unknown → skip
        )
        P, y = held_out_v1(v1, res, hosts=())
        assert len(P) == 1

    def test_skips_row_when_home_unknown(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("Mexico",))
        res = _results(("Germany", "Mexico", 1, 0))
        P, y = held_out_v1(v1, res, hosts=())
        assert len(P) == 0

    def test_p_shape_and_sums_to_one(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico", "Canada"))
        res = _results(
            ("USA", "Mexico", 1, 0),
            ("Mexico", "Canada", 0, 0),
        )
        P, y = held_out_v1(v1, res, hosts=())
        assert P.shape == (2, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_outcomes_aligned(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico", "Canada"))
        res = _results(
            ("USA", "Mexico", 2, 0),    # home win → 0
            ("Mexico", "Canada", 1, 1), # draw → 1
            ("Canada", "USA", 0, 3),    # away win → 2
        )
        _, y = held_out_v1(v1, res, hosts=())
        assert list(y) == [0, 1, 2]

    def test_empty_results(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("United States", "Mexico"))
        empty = pd.DataFrame(
            columns=["home", "away", "home_goals", "away_goals", "outcome", "stage"]
        )
        P, y = held_out_v1(v1, empty, hosts=())
        assert P.shape == (0, 3)
        assert y.shape == (0,)

    def test_default_hosts_includes_mexico(self):
        """Default hosts=(Mexico,USA,Canada); Mexico home → host_home=True."""
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("Mexico", "Poland"))
        res = _results(("Mexico", "Poland", 0, 0))
        held_out_v1(v1, res)  # default hosts
        assert v1._calls[0][2] is True

    def test_no_host_flag_when_not_in_hosts(self):
        from cupcast.compare.wc2026 import held_out_v1

        v1 = _StubV1Fit(("Germany", "France"))
        res = _results(("Germany", "France", 1, 0))
        held_out_v1(v1, res, hosts=())
        assert v1._calls[0][2] is False
        assert v1._calls[0][3] is False


# ---------------------------------------------------------------------------
# compare_on_common
# ---------------------------------------------------------------------------


class TestCompareOnCommon:
    """
    v2 knows: France, Brazil, Argentina, USA
    v1 knows: France, Brazil, United States (USA maps to United States)
    Results:
      France vs Brazil     → both can score         (common)
      USA vs France        → v2: USA known; v1: United States known  (common)
      Argentina vs France  → v2 can, v1 cannot (Argentina ∉ v1)  (NOT common)
    Common n = 2.
    """

    def _setup(self):
        v2_posterior = _stub_posterior(("France", "Brazil", "Argentina", "USA"))
        v1_fit = _StubV1Fit(("France", "Brazil", "United States"))
        results = _results(
            ("France", "Brazil", 1, 0),
            ("USA", "France", 0, 1),
            ("Argentina", "France", 2, 0),
        )
        return results, v2_posterior, v1_fit

    def test_returns_three_rows(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        assert len(df) == 3

    def test_forecaster_names(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        assert set(df["forecaster"]) == {"v2", "v1", "uniform"}

    def test_all_scored_on_same_n(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        assert df["n"].nunique() == 1

    def test_common_n_is_two(self):
        """Argentina row excluded (not in v1), so only 2 common matches."""
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        assert df["n"].iloc[0] == 2

    def test_metrics_are_finite(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        for col in ("log_loss", "brier", "rps"):
            assert np.isfinite(df[col]).all(), f"{col} has non-finite values"

    def test_columns(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        assert set(df.columns) >= {"forecaster", "n", "log_loss", "brier", "rps"}

    def test_uniform_log_loss_equals_ln3(self):
        from cupcast.compare.wc2026 import compare_on_common

        df = compare_on_common(*self._setup())
        uniform_ll = df.loc[df["forecaster"] == "uniform", "log_loss"].iloc[0]
        assert uniform_ll == pytest.approx(np.log(3), rel=1e-6)

    def test_all_results_known_to_both_n_equals_total(self):
        """When all rows are scorable by both, n equals total rows."""
        from cupcast.compare.wc2026 import compare_on_common

        v2_posterior = _stub_posterior(("France", "Brazil"))
        v1_fit = _StubV1Fit(("France", "Brazil"))
        results = _results(
            ("France", "Brazil", 1, 0),
            ("Brazil", "France", 0, 0),
        )
        df = compare_on_common(results, v2_posterior, v1_fit)
        assert df["n"].iloc[0] == 2
