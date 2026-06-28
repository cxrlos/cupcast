"""Deterministic v2 forecast report.

``main()`` is the single forecast entrypoint: it gates on API-Football/ESPN
agreement, fits the v2 model once, and regenerates every ``outputs/`` artifact —
the pre-tournament forecast, the context-adjusted match predictions, the live
bracket forecast conditioned on the actual draw, and the full projected bracket
(through the final and the third-place match).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cupcast.compare.wc2026 import V1_NAME_MAP
from cupcast.v2.context.adjust import adjusted_outcome_probs
from cupcast.v2.context.covariates import match_context
from cupcast.v2.context.venues import VENUES
from cupcast.v2.fetch import endpoints
from cupcast.v2.fetch.client import ApiFootballClient
from cupcast.v2.fetch.espn import EspnClient
from cupcast.v2.gate import verify_sources
from cupcast.v2.model import predict
from cupcast.v2.sim.knockout import advance_probability
from cupcast.v2.sim.live_bracket import resolve_live_r32, simulate_live_knockouts, validate_live_r32
from cupcast.v2.sim.monte_carlo import simulate_tournament
from cupcast.v2.sim.structure import (
    ALL_TEAMS,
    FINAL,
    GROUP_LETTERS,
    HOST_COUNTRIES,
    QUARTERFINALS,
    R16,
    R32,
    SEMIFINALS,
    TEAM_GROUP,
    THIRD_PLACE,
)

OUT = Path("outputs")
MARKET_CSV = Path("data/processed/market_outrights.csv")
CUTOFF = "2026-06-11"
SEED = 2026
N_SIMS = 50_000
STEPS = 2500
HOSTS = HOST_COUNTRIES
_CLUB_LEAGUES = (39, 61, 71, 78, 88, 94, 128, 135, 140, 144, 203, 253, 262, 307)
_INTL_FORECAST = [
    (1, 2026), (4, 2024), (9, 2024), (6, 2025), (7, 2023), (22, 2025), (5, 2024),
    (536, 2024), (29, 2023), (32, 2024), (34, 2022), (31, 2022), (10, 2025),
]
_INTL_IDS = {1, 4, 5, 6, 7, 9, 10, 22, 29, 30, 31, 32, 33, 34, 37, 536, 808}
_CITY_ALIAS = {"Guadalajara": "Zapopan"}
_R32_VENUE = {m: v for m, *_, v in R32}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def modal_score(matrix: np.ndarray) -> str:
    i, j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return f"{i}-{j}"


def podium_counts(champ_ids, runner_ids, third_ids, teams, top: int = 10) -> pd.DataFrame:
    triples = pd.DataFrame({
        "champion": [teams[i] for i in champ_ids],
        "runner_up": [teams[i] for i in runner_ids],
        "third": [teams[i] for i in third_ids],
    })
    return (triples.value_counts(normalize=True).head(top)
            .rename("probability").reset_index())


def venue_altitude(city: str) -> int | None:
    v = VENUES.get(_CITY_ALIAS.get(city, city))
    return v.elevation_m if v else None


def match_ctx(is_host: bool, altitude_m: int | None) -> dict:
    """Venue-intrinsic context for a knockout tie (travel/rest are path-free here)."""
    return {"is_host": is_host, "travel_km": 0.0, "rest_days": None, "altitude_m": altitude_m}


# --------------------------------------------------------------------------- #
# Heavy model fit
# --------------------------------------------------------------------------- #
def fit_forecast_model(client):
    import jax.numpy as jnp

    from cupcast.v2.clubform import composite
    from cupcast.v2.clubform.league_strength import (
        fetch_clubelo,
        league_strength_index,
        normalize_club_name,
    )
    from cupcast.v2.clubform.shootout import build_shootout
    from cupcast.v2.model import fit as fitmod
    from cupcast.v2.model.dixon_coles import team_index
    from cupcast.v2.model.matches import assemble_internationals
    from cupcast.v2.model.prior import clubform_prior_locs, dynamic_dc_with_prior

    c2l: dict[str, int] = {}
    for fp in glob.glob("data/raw/api_football/fixtures/*.json"):
        with open(fp) as fh:
            response = json.load(fh).get("response", [])
        for f in response:
            for s in ("home", "away"):
                c2l[normalize_club_name(f["teams"][s]["name"])] = f["league"]["id"]
    strength = league_strength_index(fetch_clubelo(), c2l)
    club = [(lid, s) for lid in _CLUB_LEAGUES for s in (2024, 2025)]
    clubform = composite.build_clubform(client, club, _INTL_FORECAST, strength)
    gkz = build_shootout(client, club, _INTL_FORECAST)

    comp_seasons = set()
    for fp in glob.glob("data/raw/api_football/fixtures/*.json"):
        with open(fp) as fh:
            r = json.load(fh).get("response", [])
        if r and r[0]["league"]["id"] in _INTL_IDS:
            comp_seasons.add((r[0]["league"]["id"], r[0]["league"]["season"]))
    m = assemble_internationals(client, sorted(comp_seasons), cutoff=CUTOFF)
    teams, hi, ai = team_index(m)
    per = m.period.to_numpy()
    ca, cd = clubform_prior_locs(clubform, teams)
    args = (jnp.asarray(hi), jnp.asarray(ai), jnp.asarray(per),
            jnp.asarray(m.host_home.to_numpy(float)), jnp.asarray(m.home_goals.to_numpy(float)),
            jnp.asarray(m.away_goals.to_numpy(float)), len(teams), int(per.max()) + 1,
            jnp.asarray(ca), jnp.asarray(cd))
    post = fitmod.fit_svi(dynamic_dc_with_prior, args, teams, seed=SEED, steps=STEPS)
    return post, gkz


# --------------------------------------------------------------------------- #
# Pre-tournament artifacts
# --------------------------------------------------------------------------- #
def write_pretournament(details) -> None:
    table = details.table
    table.round(5).to_csv(OUT / "simulation_results.csv", index=False)

    lines = ["# Top-3 forecast", ""]
    for slot, col in (("Champion", "p_champion"), ("Runner-up", "p_runner_up"),
                      ("Third place", "p_podium_third")):
        lines += [f"## {slot}", ""]
        for r in table.nlargest(5, col).itertuples():
            lines.append(f"- {r.team}: {getattr(r, col):.1%}")
        lines.append("")
    pod = podium_counts(details.match_winner[104], details.match_loser[104],
                        details.match_winner[103], ALL_TEAMS)
    lines += ["## Most likely podiums (joint)", "",
              "| Champion | Runner-up | Third | Probability |", "|---|---|---|---|"]
    for r in pod.itertuples():
        lines.append(f"| {r.champion} | {r.runner_up} | {r.third} | {r.probability:.3%} |")
    (OUT / "top3_forecast.md").write_text("\n".join(lines) + "\n")

    t = table.set_index("team")
    el = ["# Executive summary — model picks", "", "## Group stage", "",
          "| Group | Winner pick | Runner-up pick | Third (qualify odds) |", "|---|---|---|---|"]
    for letter, gf in table.groupby("group"):
        o = gf.sort_values("p_group_win", ascending=False)
        win = o.iloc[0]
        rest = gf[gf.team != win.team]
        run = rest.sort_values("p_group_runner_up", ascending=False).iloc[0]
        third = rest[rest.team != run.team].sort_values("p_qualify", ascending=False).iloc[0]
        el.append(f"| {letter} | {win.team} ({win.p_group_win:.0%}) "
                  f"| {run.team} ({run.p_group_runner_up:.0%}) "
                  f"| {third.team} ({third.p_qualify:.0%}) |")
    el += ["", "## Knockout picks (advance the higher-probability side)", ""]
    for stage, col, n in (("Quarter-finalists", "p_qf", 8), ("Semi-finalists", "p_sf", 4),
                          ("Finalists", "p_final", 2)):
        el.append(f"- **{stage}**: " + ", ".join(t.nlargest(n, col).index))
    champ = t["p_champion"].idxmax()
    run = t.drop(champ)["p_runner_up"].idxmax()
    third = t.drop([champ, run])["p_podium_third"].idxmax()
    el += ["", "## Podium", "", f"1. **{champ}** ({t.loc[champ, 'p_champion']:.1%})",
           f"2. **{run}** ({t.loc[run, 'p_runner_up']:.1%})",
           f"3. **{third}** ({t.loc[third, 'p_podium_third']:.1%})"]
    (OUT / "executive_summary.md").write_text("\n".join(el) + "\n")

    if MARKET_CSV.exists():
        market = pd.read_csv(MARKET_CSV)
        mt = table[["team", "p_champion"]].copy()
        mt["market_team"] = mt["team"].map(lambda x: V1_NAME_MAP.get(x, x))
        merged = mt.merge(market, left_on="market_team", right_on="team", how="left",
                          suffixes=("", "_m"))
        merged["edge"] = merged["p_champion"] - merged["market_p_champion"]
        merged = merged.sort_values("edge", key=lambda s: s.abs(), ascending=False)
        merged[["team", "p_champion", "market_p_champion", "edge"]].to_csv(
            OUT / "market_comparison.csv", index=False)
        ml = ["# Model vs market (championship probabilities)", "",
              "| Team | Model | Market | Edge |", "|---|---|---|---|"]
        for r in merged.head(15).itertuples():
            mp = "-" if pd.isna(r.market_p_champion) else f"{r.market_p_champion:.1%}"
            ml.append(f"| {r.team} | {r.p_champion:.1%} | {mp} | {r.edge:+.1%} |")
        (OUT / "market_comparison.md").write_text("\n".join(ml) + "\n")

    tid = {team: i for i, team in enumerate(ALL_TEAMS)}
    champ_ids = details.match_winner[104]
    rows = []
    for team in table.nlargest(8, "p_champion")["team"]:
        i = tid[team]
        g = GROUP_LETTERS.index(TEAM_GROUP[team])
        is_ch = champ_ids == i
        masks = {"p_champ_if_group_win": details.winners[g] == i,
                 "p_champ_if_runner_up": details.runners[g] == i,
                 "p_champ_if_third": (details.thirds[g] == i) & details.qualifies[g]}
        row = {"team": team, "p_champion": round(float(is_ch.mean()), 4)}
        for col, mask in masks.items():
            row[col] = round(float(is_ch[mask].mean()), 4) if mask.any() else float("nan")
        rows.append(row)
    sens = pd.DataFrame(rows)
    (OUT / "sensitivity.md").write_text(
        "\n".join(["# Sensitivity analysis", "",
                   "## Path dependence: P(champion | group-stage outcome)", "",
                   sens.to_markdown(index=False), "",
                   "Conditioning is over the same 50k simulations; finishing third forces a",
                   "tougher bracket path, which is the gap between the columns.", ""]) + "\n")


# --------------------------------------------------------------------------- #
# Match predictions (context-adjusted) + ESPN venue resolution
# --------------------------------------------------------------------------- #
def espn_r32_venues(espn: EspnClient) -> dict[frozenset, str]:
    out: dict[frozenset, str] = {}
    for fx in espn.scheduled_fixtures("20260628", "20260704"):
        if "winner" in fx["home"] or "winner" in fx["away"]:
            continue
        out[frozenset({fx["home"], fx["away"]})] = fx.get("venue_city", "")
    return out


def _canon_pair(a: str, b: str) -> frozenset:
    from cupcast.v2.fetch.espn import canon
    return frozenset({canon(a), canon(b)})


def write_match_predictions(post, gkz, fixtures, resolved, venue_by_pair) -> None:
    known = set(post.teams)
    grp = [f for f in fixtures if "Group Stage" in str(f["league"].get("round", ""))]
    grows = []
    for f in sorted(grp, key=lambda f: (TEAM_GROUP.get(f["teams"]["home"]["name"], "Z"),
                                        f["fixture"]["date"])):
        h, a = f["teams"]["home"]["name"], f["teams"]["away"]["name"]
        if h not in known or a not in known:
            continue
        ch = match_context(grp, h, f)
        ca = match_context(grp, a, f)
        (ph, pdr, pa), _ = adjusted_outcome_probs(post, h, a, ch, ca)
        egh, ega = predict.expected_goals(post, h, a, h in HOSTS, False)
        gmat = predict.score_matrix(post, h, a, h in HOSTS, False)
        grows.append({"group": TEAM_GROUP.get(h, "?"), "home": h, "away": a,
                      "p_home": round(ph, 4), "p_draw": round(pdr, 4), "p_away": round(pa, 4),
                      "xg_home": round(egh, 2), "xg_away": round(ega, 2),
                      "modal_score": modal_score(gmat)})
    gdf = pd.DataFrame(grows)
    gdf.to_csv(OUT / "match_predictions_groups.csv", index=False)

    krows = []
    for slot in sorted(resolved):
        h, a = resolved[slot]
        vc = _R32_VENUE[slot]
        city = venue_by_pair.get(_canon_pair(h, a), "")
        hh, ha = h in HOSTS and vc == h, a in HOSTS and vc == a
        ch = match_ctx(hh, venue_altitude(city))
        ca = match_ctx(ha, venue_altitude(city))
        (ph, pdr, pa), _ = adjusted_outcome_probs(post, h, a, ch, ca)
        adv = advance_probability(post, h, a, vc if vc in HOSTS else "", gkz)
        egh, ega = predict.expected_goals(post, h, a, hh, ha)
        krows.append({"match": slot, "home": h, "away": a, "p_home": round(ph, 4),
                      "p_draw": round(pdr, 4), "p_away": round(pa, 4),
                      "p_home_advances": round(adv, 4), "xg_home": round(egh, 2),
                      "xg_away": round(ega, 2),
                      "modal_score": modal_score(predict.score_matrix(post, h, a, hh, ha))})
    kdf = pd.DataFrame(krows)
    kdf.to_csv(OUT / "match_predictions_knockouts.csv", index=False)

    lines = ["# Match predictions", "",
             "Model trained to the 11 June 2026 cutoff; probabilities home/draw/away, "
             "context layer (host/altitude/travel) applied where venue data is available.", ""]
    for g, sub in gdf.groupby("group"):
        lines += [f"## Group {g}", "", "| Match | P(H/D/A) | xG | Modal |", "|---|---|---|---|"]
        for r in sub.itertuples():
            lines.append(f"| {r.home} v {r.away} | {r.p_home:.2f}/{r.p_draw:.2f}/{r.p_away:.2f} "
                         f"| {r.xg_home:.2f}-{r.xg_away:.2f} | {r.modal_score} |")
        lines.append("")
    lines += ["## Round of 32 (actual bracket)", "",
              "| Match | P(H/D/A) | P(home advances) | xG | Modal |", "|---|---|---|---|---|"]
    for r in kdf.itertuples():
        lines.append(f"| {r.home} v {r.away} "
                     f"| {r.p_home:.2f}/{r.p_draw:.2f}/{r.p_away:.2f} "
                     f"| {r.p_home_advances:.2f} "
                     f"| {r.xg_home:.2f}-{r.xg_away:.2f} | {r.modal_score} |")
    (OUT / "match_predictions.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Live conditioned forecast + full projected bracket (incl. third place)
# --------------------------------------------------------------------------- #
def write_live_forecast(post, gkz, resolved) -> None:
    tbl = simulate_live_knockouts(post, resolved, gk_z=gkz, n_sims=N_SIMS, seed=SEED)
    tbl.to_csv(OUT / "knockout_bracket_forecast.csv", index=False)
    lines = ["# Cupcast II — Knockout bracket forecast (conditioned on the actual draw)", "",
             f"From {N_SIMS:,} Monte Carlo simulations seeded with the real Round-of-32 bracket, "
             "with the penalty shootout edge. Seed 2026.", "",
             "| Team | Champion | Final | Semi | QF |", "|---|---:|---:|---:|---:|"]
    for r in tbl.head(16).itertuples():
        lines.append(f"| {r.team} | {r.p_champion:.1%} | {r.p_final:.1%} "
                     f"| {r.p_sf:.1%} | {r.p_qf:.1%} |")
    (OUT / "knockout_bracket_forecast.md").write_text("\n".join(lines) + "\n")


def project_bracket(post, gkz, resolved, advance_fn=advance_probability) -> dict[int, dict]:
    """Single most-likely bracket: at each tie the higher advance-probability side wins.

    Covers R32 -> R16 -> QF -> SF -> third-place (M103) -> final (M104).
    """
    winner: dict[int, str] = {}
    loser: dict[int, str] = {}
    record: dict[int, dict] = {}

    def play(slot: int, a: str, b: str, venue: str) -> None:
        adv = advance_fn(post, a, b, venue if venue in HOSTS else "", gkz)
        w, p = (a, adv) if adv >= 0.5 else (b, 1 - adv)
        winner[slot], loser[slot] = w, (b if w == a else a)
        record[slot] = {"home": a, "away": b, "winner": w, "p": p, "venue": venue}

    for slot in sorted(resolved):
        a, b = resolved[slot]
        play(slot, a, b, _R32_VENUE[slot])
    for stage in (R16, QUARTERFINALS, SEMIFINALS):
        for slot, fa, fb, venue in stage:
            play(slot, winner[fa], winner[fb], venue)
    tm, s1, s2, tv = THIRD_PLACE
    play(tm, loser[s1], loser[s2], tv)
    fm, f1, f2, fv = FINAL
    play(fm, winner[f1], winner[f2], fv)
    return record


def bracket_results_md(record: dict[int, dict]) -> str:
    def row(slot: int) -> str:
        r = record[slot]
        return f"| {r['home']} vs {r['away']} | **{r['winner']}** | {r['p']:.0%} |"

    lines = ["# Cupcast II — Full projected bracket (most-likely path)", "",
             "The single most-likely bracket: at each tie the higher advance-probability "
             "side (regulation + extra time + penalty shootout) wins. "
             "Probabilities are that tie only.", ""]
    for title, slots in (("Round of 32", [m for m, *_ in R32]),
                         ("Round of 16", [m for m, *_ in R16]),
                         ("Quarter-finals", [m for m, *_ in QUARTERFINALS]),
                         ("Semi-finals", [m for m, *_ in SEMIFINALS])):
        lines += [f"## {title}", "", "| Tie | Advances | P |", "|---|---|---:|"]
        lines += [row(s) for s in slots]
        lines.append("")
    tm = THIRD_PLACE[0]
    fm = FINAL[0]
    lines += ["## Third-place match", "", "| Tie | Winner | P |", "|---|---|---:|", row(tm), ""]
    lines += ["## Final", "", "| Tie | Champion | P |", "|---|---|---:|", row(fm), "",
              f"**Projected champion: {record[fm]['winner']}** · "
              f"**Third place: {record[tm]['winner']}**"]
    return "\n".join(lines) + "\n"


def write_bracket_results(record: dict[int, dict]) -> None:
    (OUT / "bracket_results.md").write_text(bracket_results_md(record))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    import os
    import warnings

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warnings.filterwarnings("ignore")
    from dotenv import load_dotenv

    load_dotenv()
    OUT.mkdir(exist_ok=True)

    af = ApiFootballClient()
    espn = EspnClient()
    print("gate: verifying API-Football and ESPN agree ...")
    verify_sources(af, espn)
    print("  sources agree.")

    post, gkz = fit_forecast_model(af)

    details = simulate_tournament(post, n_sims=N_SIMS, seed=SEED, gk_z=gkz)
    write_pretournament(details)

    fixtures = endpoints.fetch_fixtures(af, 1, 2026)
    standings = af.get_response("standings", {"league": 1, "season": 2026})
    winners, runners = {}, {}
    for lg in standings:
        for grouping in lg["league"]["standings"]:
            for r in grouping:
                g = r["group"].replace("Group ", "").strip()
                if g not in GROUP_LETTERS:
                    continue
                if r["rank"] == 1:
                    winners[g] = r["team"]["name"]
                elif r["rank"] == 2:
                    runners[g] = r["team"]["name"]
    r32_fx = [f for f in fixtures if str(f["league"].get("round", "")) == "Round of 32"]
    resolved = resolve_live_r32(winners, runners, r32_fx)
    report = validate_live_r32(resolved, r32_fx)
    if not report["ok"]:
        raise RuntimeError(f"live bracket invalid: {report['issues']}")

    write_match_predictions(post, gkz, fixtures, resolved, espn_r32_venues(espn))
    write_live_forecast(post, gkz, resolved)
    write_bracket_results(project_bracket(post, gkz, resolved))
    print("wrote outputs/: simulation_results, top3_forecast, executive_summary,")
    print("  market_comparison, sensitivity, match_predictions, knockout_bracket_forecast,")
    print("  bracket_results")


if __name__ == "__main__":
    main()
