# Context Layer Implementation Plan

Build `cupcast.v2.context`: a bounded, audited context layer that adjusts WC2026 match
predictions for structured, computable signals — travel/rest, venue/crowd/host (round-aware),
and altitude — with each effect hard-capped so it can never dominate the statistical core,
plus a shadow-mode hook for LLM-extracted morale (extracted/logged, zero weight until proven).
The pre/post-context ablation (Plan 7 Task 4) then measures the layer's marginal value on the
held-out WC2026 results.

**Architecture:** `venues.py` (a static table of the 16 host venues: coordinates, elevation,
country, host nation, and how late each hosts), `covariates.py` (per-match structured covariates
computed from the cached WC2026 schedule + the venue table — great-circle travel distance, rest
days, host/crowd flag, altitude gap), `adjust.py` (bounded, transparent prediction adjustments
applied on top of `predict.outcome_probs`/`score_matrix`, each clipped to a small max effect),
and a `morale.py` shadow stub (interface only; zero weight). Effects are conservative,
hand-set caps (per the design's "bounded, can't dominate"), audited per match.

**Tech Stack:** Python ≥3.12, numpy/pandas, pytest. Structured-first; no model re-fit required —
the layer adjusts predictions, so it composes with the existing `predict`/`sim`.

## Locked design decisions (from the brainstorm + source analysis)

- 4 of 6 covariate families are **computed** (travel/rest, venue/crowd/altitude, weather, injury
  spine) — deterministic, zero alarmism. This plan does travel/rest + venue/crowd/altitude
  (weather + injuries: follow-on tasks). Morale runs in **shadow mode** (zero weight).
- **Venue facts:** Mexico hosts through the Round of 32 (Azteca/Mexico City ≈2240 m,
  Akron/Guadalajara ≈1560 m, BBVA/Monterrey ≈540 m); from the quarter-finals every match is in the
  US. Host advantage is therefore **venue- and round-aware**, plus a smaller, capped
  "traveling-support" effect for huge pro-team crowds at US venues.
- **Bounded & audited:** every covariate maps to a capped strength nudge (e.g. ≤ a few % on a
  match outcome), with a per-match provenance record (which covariate, raw value, capped applied
  value). Math owns the probabilities.

## Global Constraints

- **NEVER `git commit`/`git push`** unless explicitly told. Stage with `git add`.
- **v2 code must NEVER import `cupcast.v1`** (clean-room guard).
- **Every stochastic step seeded.** Package/paths lowercase `cupcast`. Read only from cache.
- Tests use small in-memory inputs; no network/key. Full suite green after each task (starts at 373).

---

### Task 1: Venue table + structured covariates (travel/rest, host/crowd, altitude)

**Files:** Create `src/cupcast/v2/context/__init__.py`, `src/cupcast/v2/context/venues.py`,
`src/cupcast/v2/context/covariates.py`. Tests `tests/v2/test_context_covariates.py`.

**Interfaces:**
- `venues.VENUES: dict[str, Venue]` — the 16 host cities keyed by the API-Football venue city
  string, each `Venue(city, country, lat, lon, elevation_m, host_nation, hosts_through_round)`.
  (host_nation ∈ {"Mexico","USA","Canada"} for those cities; `hosts_through_round` e.g. "R32" for
  Mexican/Canadian venues, "Final" for US venues.) `venue_for(city) -> Venue | None`.
- `covariates.great_circle_km(lat1, lon1, lat2, lon2) -> float`.
- `covariates.match_context(fixtures: list[dict], team: str, fixture: dict) -> dict` — for a team
  in a given WC2026 fixture, compute from the team's schedule (the cached fixtures): `rest_days`
  (days since its previous match), `travel_km` (great-circle from its previous venue to this one),
  `is_host` (team's nation == this venue's host_nation), `altitude_m` (venue elevation),
  `altitude_gap_m` (this venue's elevation minus the team's typical/home elevation, default 0 if
  unknown). Returns the raw covariate dict (uncapped) + the venue.
- `covariates.team_schedule(fixtures, team) -> list[dict]` — that team's WC2026 fixtures sorted by date.

- [ ] **Step 1:** failing tests — `great_circle_km` matches a known distance (e.g. Mexico City→Monterrey ≈ 700 km, within tolerance); `venue_for` resolves a known city and returns its host_nation/elevation; `match_context` on a small in-memory two-match schedule computes the right `rest_days`, `travel_km` (0 for the first match), and `is_host` flag (a Mexican team at a Mexican venue → True; same team at a US venue → False).
- [ ] **Step 2–6:** fail → implement → pass → suite/lint/clean-room → stage. **Controller real check:** compute `match_context` for a few real WC2026 fixtures (e.g. Iran's group matches → realistic travel/rest; a Mexico home match → is_host True, altitude≈2240) and sanity-print.

### Task 2: Bounded prediction adjustments + provenance

**Files:** Create `src/cupcast/v2/context/adjust.py`. Test `tests/v2/test_context_adjust.py`.

**Interfaces:**
- `CAPS` — per-covariate max effect on the log-rate (small, conservative; e.g. host `≤ +0.12`,
  travel fatigue `≤ -0.10`, short rest `≤ -0.08`, altitude gap `≤ -0.10` for the non-adapted side),
  documented and citable.
- `rate_multipliers(ctx_home: dict, ctx_away: dict) -> tuple[float, float, list[dict]]` — convert
  each side's capped covariates into a multiplicative adjustment on its scoring rate
  (`exp(sum of capped log-effects)`), returning `(mult_home, mult_away, provenance)` where
  provenance lists each covariate's raw + capped applied value.
- `adjusted_outcome_probs(posterior, home, away, ctx_home, ctx_away, host_home, host_away) ->
  tuple[probs, provenance]` — apply the multipliers via `predict.score_matrix(..., rate_scale=...)`
  per side (or a small extension accepting separate home/away scales) and return W/D/L + the audit.

- [ ] **Step 1:** failing tests — each covariate clipped to its cap (an extreme raw value never
  exceeds the cap); a host team's adjusted win prob rises but by a BOUNDED amount; provenance
  records every covariate; with all covariates zero, `adjusted_outcome_probs` == the base
  `outcome_probs`. (stub posterior)
- [ ] **Step 2–6:** fail → implement → pass → suite/lint/clean-room → stage.

### Task 3: Morale shadow stub + context-aware simulation hook

**Files:** Create `src/cupcast/v2/context/morale.py` (shadow stub) and extend the sim/predict path
to optionally accept per-match context. Test `tests/v2/test_context_shadow.py`.

**Interfaces:**
- `morale.morale_signal(team, as_of) -> dict` — interface returning `{value: 0.0, weight: 0.0,
  sources: []}` (SHADOW: always zero weight; real extraction wired later). Logged, never applied.
- A `context_provider` callable the simulator can take so knockout/group predictions are computed
  via `adjusted_outcome_probs` when context is supplied, falling back to the base predictions when
  not (keeps the existing sim behavior the default).

- [ ] **Step 1:** failing tests — `morale_signal` returns zero weight (shadow); the sim with a
  zero-effect context provider produces IDENTICAL results to the sim without one (the ablation
  baseline); a non-zero host context provider shifts a host team's advancement up but bounded.
- [ ] **Step 2–6:** fail → implement → pass → suite/lint/clean-room → stage. **Controller real
  check:** run the WC2026 sim WITH the structured context vs WITHOUT; print the championship-prob
  deltas (sanity: host nations + low-travel teams nudge up slightly, none dominate).

---

## Self-Review

- **Spec coverage (design §5, §8 context layer):** the structured, computable covariates
  (travel/rest, venue/crowd/altitude — the Iran-travel and Mexico-crowd cases) with bounded,
  audited effects (Tasks 1–2), the shadow-mode morale hook + the context-aware sim path enabling
  the pre/post ablation (Task 3). Weather (Open-Meteo) and the injuries/lineup-intent + real LLM
  extraction are scoped as follow-on tasks; the structured bulk + the ablation hook are here.
- **Bounded & honest:** every effect is hard-capped and provenance-logged; the math owns the
  probabilities; morale is zero-weight until the ablation proves it.
- **No re-fit:** the layer adjusts predictions, so it composes with the existing model/sim and the
  pre/post-context ablation (Plan 7 Task 4) measures its marginal value on the WC2026 hold-out.
