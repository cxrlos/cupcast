"""Two-source verification gate: API-Football vs ESPN agreement checks."""

from __future__ import annotations

from dataclasses import dataclass

from cupcast.v2.fetch.espn import canon


class ReconcileError(RuntimeError):
    pass


@dataclass
class ReconcileReport:
    ok: bool
    matched: int
    mismatches: list
    missing: list
    label: str


def reconcile_scores(
    api_results: list[dict],
    espn_results: list[dict],
) -> ReconcileReport:
    """Compare completed-match scores between API-Football and ESPN.

    Each result dict must have keys: home, away, home_goals, away_goals.
    Names are normalized via canon() before matching.
    """
    def _index(results: list[dict]) -> dict[frozenset, dict]:
        idx: dict[frozenset, dict] = {}
        for r in results:
            key = frozenset({canon(r["home"]), canon(r["away"])})
            idx[key] = r
        return idx

    api_idx = _index(api_results)
    espn_idx = _index(espn_results)

    mismatches: list = []
    missing: list = []
    matched = 0

    all_keys = set(api_idx) | set(espn_idx)
    for key in all_keys:
        in_api = key in api_idx
        in_espn = key in espn_idx
        if in_api and in_espn:
            a = api_idx[key]
            e = espn_idx[key]
            # Scores may be stored home/away relative to each source separately;
            # normalise by sorting so we compare the same perspective.
            a_pair = _sorted_score(a)
            e_pair = _sorted_score(e)
            if a_pair != e_pair:
                mismatches.append(
                    {
                        "teams": sorted(key),
                        "api": a_pair,
                        "espn": e_pair,
                    }
                )
            else:
                matched += 1
        elif in_api:
            missing.append({"source": "espn", "teams": sorted(key)})
        else:
            missing.append({"source": "api", "teams": sorted(key)})

    ok = not mismatches and not missing
    return ReconcileReport(
        ok=ok,
        matched=matched,
        mismatches=mismatches,
        missing=missing,
        label="scores",
    )


def _sorted_score(r: dict) -> tuple[int, int]:
    """Return goals as (lower, higher) so home/away orientation doesn't matter."""
    a, b = int(r["home_goals"]), int(r["away_goals"])
    return (min(a, b), max(a, b))


def reconcile_matchups(
    our: dict[int, tuple[str, str]],
    espn_pairs: list[frozenset[str]],
) -> ReconcileReport:
    """Verify every slot matchup we computed appears in ESPN's live fixture list.

    our: slot_id → (team_a, team_b) with API-Football spellings
    espn_pairs: frozenset of two canon() tokens per ESPN fixture
    """
    espn_set = set(espn_pairs)
    mismatches: list = []
    missing: list = []
    matched = 0

    seen_espn: set[frozenset] = set()
    for slot_id, (t1, t2) in our.items():
        key = frozenset({canon(t1), canon(t2)})
        if key in espn_set:
            matched += 1
            seen_espn.add(key)
        else:
            mismatches.append({"slot": slot_id, "teams": sorted(key), "reason": "not in espn"})

    for pair in espn_set:
        if pair not in seen_espn:
            missing.append({"source": "our", "teams": sorted(pair)})

    ok = not mismatches and not missing
    return ReconcileReport(
        ok=ok,
        matched=matched,
        mismatches=mismatches,
        missing=missing,
        label="matchups",
    )


def reconcile_standings(
    api: dict[str, list[dict]],
    espn: dict[str, list[dict]],
) -> ReconcileReport:
    """Compare per-group rank order between API-Football and ESPN standings.

    Each source is a dict: group_letter → [{"team", "rank", ...}] rank-sorted.
    Only the rank-ordered team sequence is compared, not raw point values.
    """
    mismatches: list = []
    missing: list = []
    matched = 0

    all_groups = set(api) | set(espn)
    for group in sorted(all_groups):
        in_api = group in api
        in_espn = group in espn
        if not in_api:
            missing.append({"source": "api", "group": group})
            continue
        if not in_espn:
            missing.append({"source": "espn", "group": group})
            continue
        api_order = [canon(r["team"]) for r in api[group]]
        espn_order = [canon(r["team"]) for r in espn[group]]
        if api_order != espn_order:
            mismatches.append(
                {"group": group, "api": api_order, "espn": espn_order}
            )
        else:
            matched += 1

    ok = not mismatches and not missing
    return ReconcileReport(
        ok=ok,
        matched=matched,
        mismatches=mismatches,
        missing=missing,
        label="standings",
    )


def assert_sources_agree(*reports: ReconcileReport) -> None:
    """Raise ReconcileError with a readable breakdown if any report is not ok."""
    failures = [r for r in reports if not r.ok]
    if not failures:
        return
    lines = ["Sources disagree — cannot proceed:"]
    for r in failures:
        lines.append(f"\n  [{r.label}] matched={r.matched}")
        for m in r.mismatches:
            lines.append(f"    MISMATCH {m}")
        for m in r.missing:
            lines.append(f"    MISSING  {m}")
    raise ReconcileError("\n".join(lines))
