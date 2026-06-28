"""Tests for the v2 report pipeline's pure helpers and bracket projection."""

from __future__ import annotations

import numpy as np

from cupcast.v2 import report
from cupcast.v2.sim.structure import FINAL, QUARTERFINALS, R16, R32, SEMIFINALS, THIRD_PLACE


def test_modal_score_picks_argmax_cell():
    mat = np.zeros((4, 4))
    mat[2, 1] = 0.9
    assert report.modal_score(mat) == "2-1"


def test_podium_counts_columns_and_top_row():
    teams = ["X", "Y", "Z"]
    pod = report.podium_counts([0, 0, 1], [1, 1, 0], [2, 2, 2], teams, top=2)
    assert list(pod.columns) == ["champion", "runner_up", "third", "probability"]
    assert pod.iloc[0]["champion"] == "X" and pod.iloc[0]["third"] == "Z"
    assert abs(pod.iloc[0]["probability"] - 2 / 3) < 1e-9


def test_venue_altitude_known_alias_and_unknown():
    assert report.venue_altitude("Mexico City") == 2240
    assert report.venue_altitude("Guadalajara") == 1560  # alias -> Zapopan
    assert report.venue_altitude("Nowhereville") is None


def test_match_ctx_shape():
    ctx = report.match_ctx(True, 2240)
    assert ctx == {"is_host": True, "travel_km": 0.0, "rest_days": None, "altitude_m": 2240}


def _stub_advance(_post, a, b, _venue, _gkz):
    """Deterministic: the alphabetically-smaller team always advances."""
    return 1.0 if a < b else 0.0


def test_project_bracket_resolves_full_tree_with_third_place():
    resolved = {m: (f"A{m:02d}", f"B{m:02d}") for m, *_ in R32}
    record = report.project_bracket(None, None, resolved, advance_fn=_stub_advance)

    expected_slots = (
        {m for m, *_ in R32} | {m for m, *_ in R16} | {m for m, *_ in QUARTERFINALS}
        | {m for m, *_ in SEMIFINALS} | {THIRD_PLACE[0], FINAL[0]}
    )
    assert set(record) == expected_slots
    assert len(record) == 32
    # third-place and final are both decided, and from the two semifinal losers/winners
    assert record[THIRD_PLACE[0]]["winner"]
    assert record[FINAL[0]]["winner"]
    assert record[THIRD_PLACE[0]]["winner"] != record[FINAL[0]]["winner"]


def test_bracket_results_md_reports_champion_and_third():
    resolved = {m: (f"A{m:02d}", f"B{m:02d}") for m, *_ in R32}
    record = report.project_bracket(None, None, resolved, advance_fn=_stub_advance)
    md = report.bracket_results_md(record)
    assert "## Third-place match" in md
    assert "Projected champion:" in md and "Third place:" in md
    assert md.endswith("\n")
