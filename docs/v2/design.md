# Cupcast v2 — Design

Status: approved design, pre-implementation. Date: 2026-06-25.

v1 is frozen as the cheap/simple baseline (kept runnable, untouched). v2 is a single
powerful model for the full tournament — group stage → Round of 32 → … → final +
third-place — built clean-room. The cheap-vs-powerful contrast becomes the v1-vs-v2
comparison paper.

This document is the contract for the implementation plan. Every decision below is
locked unless explicitly revisited.

---

## 1. Goals and non-goals

### Goals
- A reproducible, scientifically defensible World-Cup-2026 forecasting model that
  measurably beats v1 and is competitive with the bookmaker market on proper scoring
  rules, evaluated out-of-sample on the actual 2026 results.
- First-class treatment of **form** (time-varying team strength), **player/lineup
  quality**, and **style/"system-vs-system" matchups** — the three things v1 cannot see.
- A bounded, audited **context layer** (injuries, travel/rest, venue/crowd/altitude,
  weather, morale) that can nudge — never dominate — the statistical core.
- An **overnight, reproducible auto-tuning** loop over a principled hyperparameter
  surface, with the 2026 tournament held out as the final test set.
- Papers with full mathematical annotation, including a v1-vs-v2 comparison and a
  formal pre-vs-post-LLM ablation.

### Non-goals (YAGNI)
- No market-value / Transfermarkt data anywhere (repo hard rule: performance metrics only).
- No fine-tuning of an LLM on match outcomes — the sample is far too small and an LLM
  has no calibrated notion of a scoreline. The statistics own all probabilities.
- No premium event data (StatsBomb/Opta) in v1 of this build — deferred behind a
  measured trigger (see §3).
- No deployed two-tier (cheap-groups / powerful-knockouts) runtime — one model end to end.

---

## 2. Why v1 fell short (diagnosis, for the comparison paper)

Evidence from the v1 repo, not speculation:
- **The player layer is inert.** `squad_strength.py` builds a single attack-only proxy
  (z-scored npxG+xA per-90, minutes-weighted, top-league coverage only). The sensitivity
  table shows "Rodri out" moves Spain's title odds by −0.05% — player identity is
  effectively invisible. v1 is Elo + static Dixon-Coles on international results with a
  cosmetic squad tilt.
- **Structural market disagreement.** v1 had Argentina 16.3% vs market 8.0% (+8.2) and
  France 7.1% vs 14.6% (−7.5) — the signature of a results-only model overweighting recent
  competitive form and underweighting squad depth.
- **Marginal tournament skill.** Euro-2024 replay log-loss 0.990 vs 1.099 uniform; club
  tier worse than Pinnacle closing (0.991 vs 0.965).
- **Static.** Frozen as-of training, no in-tournament updating, fixed 2.5y exponential
  decay as the only nod to recency.

v2 targets each of these directly: dynamic strength (form), a learned lineup-aware prior
and style matchup term (player/system), and in-tournament filtering (recency).

---

## 3. Data layer

- **Backbone: API-Football (Ultra tier, 75k requests/day).** All competitions incl. local
  leagues, ≥2 years history, national teams + qualifiers + continental + friendlies + clubs,
  lineups, injuries, per-fixture player stats, and the in-tournament results feed. Reuses the
  existing cache-first client + request ledger (hard rule: never bypass the ledger). The daily
  quota resets, so the fetch layer **maximizes data per request and never re-requests cached
  data** — bulk pulls, persisted once, to stay inside 75k/day (if a day's quota is exhausted,
  resume the next day). Key lives in `.env` as `API_FOOTBALL_KEY`; never committed or printed.
- **xG: Understat (free)**, via the already-vendored `soccerdata`, for top-league shot
  quality.
- **Premium event data (StatsBomb/Opta): deferred.** Added **only if** the WC2026 ablation
  shows the model is missing shot-quality / tactical-style signal. Budget ceiling ~$40/mo;
  the entire context layer adds **$0** (all sources free).
- All clients stay cache-first under `data/raw/`; `data/` is never committed.

---

## 4. Repository re-scaffold

```
src/cupcast/
  v1/        frozen snapshot of today's code, still runnable (the baseline)
  v2/        new, clean-room; never imports v1
  compare/   shared v1-vs-v2 evaluation harness
docs/
  v1/        existing three papers (moved, unchanged)
  v2/        new papers + this design doc
```

One repo, one `uv` environment. v2 imports nothing from v1 so the baseline stays an honest
"before". `compare/` is the only place that touches both.

### v2 package layout
```
src/cupcast/v2/
  fetch/      cache-first clients (API-Football extended: lineups, injuries, player stats)
  clubform/   2-yr club player form + role/formation/style profiles, league-adjusted
  features/   match table, squad/lineup features, expected minutes, context covariates
  context/    LLM bounded-covariate extractor + source whitelist + provenance log
  model/      dynamic state-space Dixon-Coles (NumPyro) + hybrid learned functions
  ratings/    Elo/Massey/Pi baselines (via penaltyblog) for the prior + baselines
  sim/        48-team bracket Monte Carlo (group → R32 → … → final + third)
  tune/       Optuna rolling-origin CV auto-tuner
  validate/   scoring rules, calibration, backtests, the pre/post-LLM ablation
  report/     outputs/ artifacts + docs/v2 figures + LaTeX prose
  run.py      orchestration
```

---

## 5. `clubform` / `players` subsystem (separate semantic)

Two years of club-level player data as a **distinct subsystem** with its own schema,
decoupled from the international goal model; the two meet only at the learned
prior/matchup interface (§6).

- Inputs: minutes by position/role, formation context, and style profile (shot locations,
  set-piece reliance, scoring patterns at the ~$40/mo tier; deep tactical fingerprints
  deferred with event data).
- **Guardrails (or this adds noise, not signal):**
  1. **League-strength normalization** — 0.5 npxG/90 in a weak league ≠ in a strong one.
  2. **Club→country transfer weight is learned and validated, never assumed** (it is the
     hybrid function in §6).
  3. **No double-counting** — "form" must add new information (trajectory, role change),
     not re-inject a number the squad prior already uses.
- Built to **accept richer event data later** (socceraction VAEP/xT, kloppy parsing)
  without re-architecting.

---

## 6. Model core — dynamic state-space Dixon-Coles → hybrid

We adopt the **inference engine (NumPyro)** and write the **model** ourselves; no library
ships a time-varying Dixon-Coles.

### 6.1 Likelihood (per match m, time t, home i, away j)
```
λ_m = exp( μ + att_i(t) − def_j(t) + γ·host_im + g_θ(style_i, style_j) + Σ_k β_k x_{m,k} )
ν_m = exp( μ + att_j(t) − def_i(t) + γ·host_jm + g_θ(style_j, style_i) + Σ_k β_k x_{m,k} )
```
Goals follow a bivariate-Poisson / independent-Poisson model with the **Dixon-Coles
low-score correction** τ(·, ρ) retained for calibration (the v1 correction is sound and
kept). `host_im` is venue-and-round-aware (§8).

### 6.2 State evolution (this is "form")
Each team's log-attack and log-defense are latent states following a Gaussian random walk:
```
att_i(t+1) = att_i(t) + ε^a_i,   ε^a_i ~ N(0, σ_att²)
def_i(t+1) = def_i(t) + ε^d_i,   ε^d_i ~ N(0, σ_def²)
```
`σ_att², σ_def²` (the **state innovation variances**) are the principled replacement for
v1's fixed decay half-life: tuned, not guessed. In-tournament updating is automatic — as
2026 matches are observed, the filter advances the latent states forward, so recency
weighting falls out of the dynamics.

### 6.3 Lineup-aware prior level (the hybrid graduation)
A team's initial / mean-reversion strength level is a **learned function** of
club-form/player features from §5:
```
att_i(0) ~ N( f_θ(squad_features_i), τ_prior² )
```
`f_θ` and the style-matchup term `g_θ` are learned via **pyGAM** (interpretable, monotone-
constrainable, defensible in the paper) and/or **LightGBM** (predictive power) — selected
empirically by rolling-origin CV, not assumed up front.

### 6.4 Context covariates `x_{m,k}`
Travel/rest, venue/crowd/altitude, weather, injuries (§8). Each coefficient `β_k` is
**capped** so no single signal can dominate; morale runs in shadow mode (β = 0 until proven).

### 6.5 Inference
- **Final model:** full NUTS (NumPyro), seeded via JAX `PRNGKey` with a deterministic
  key-threading harness for exact reproducibility. ArviZ for R-hat/ESS/posterior-predictive
  diagnostics (validation-paper figures).
- **Tuning trials:** fast variational / Laplace fits inside the Optuna loop (§11) so the
  overnight budget buys many trials; the selected config is re-fit with full NUTS.

---

## 7. Simulation

Monte Carlo over the official 48-team format, **built ourselves** (no library encodes it):
12 groups → top-2 + 8 best thirds = **Round of 32** → R16 → QF → SF → **final + third-place**.
FIFA group tiebreaker cascade, extra time as rescaled rates, penalty shootouts with a small
GK-quality adjustment. Pure NumPy, 50k sims, fully seeded. The bracket structure is built to
the official template now and populated once the real bracket is confirmed.

---

## 8. Context layer (LLM = bounded extractor + reporter)

**Structured-first, prose-gated, effect-capped.** Four of six covariate families are
**computed from free structured sources** (zero alarmism); the LLM is reserved for the two
that genuinely need text understanding plus report prose.

| Family | Source / method | LLM role |
|---|---|---|
| Travel / rest / logistics | FIFA schedule + Wikidata venue coords → great-circle distance, time-zone delta, rest days | none (computed); LLM only flags rare corroborated late-arrival events |
| Venue / crowd / altitude / climate | host status (venue+round-aware), Wikidata coords, Open-Meteo elevation/heat | none (computed) |
| Weather | Open-Meteo (no key, free) | none (computed) |
| Injuries / suspensions | API-Football injuries endpoint + card accumulation | confirm/upgrade certainty only; never fabricate an absence |
| Tactical / lineup intent | LLM from whitelist | small-cap covariate |
| Morale / turmoil | LLM from whitelist | **shadow mode** — extracted, logged, shown in reports, **β = 0** until the ablation proves it earns activation |

### Venue / crowd specifics
Mexico's three venues (Azteca/Mexico City ~2,240 m, Akron/Guadalajara, BBVA/Monterrey)
host through the **Round of 32**; from the **quarter-finals onward every match is in the
US**. So the true home-venue term applies only through R32, with a **separate, smaller,
bounded traveling-support effect** for large pro-team crowds in US venues. Altitude and heat
enter as computed covariates.

### Source whitelist (all free)
- **Structured backbone:** API-Football, FIFA schedule, Open-Meteo, Wikidata.
- **Text:** Reuters + AP wires, BBC Sport, federation/confederation official channels
  (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC + national FAs), and a breadth layer of
  GDELT (whitelisted-domain corroboration **counter** only, never raw tone), Guardian, ESPN.

### Filtering rules
1. Source hierarchy with hard precedence: structured/official > wire (Reuters/AP) >
   reputable outlet (BBC/Guardian/ESPN) > rejected. Injuries must originate structured/
   official; prose may only upgrade certainty.
2. Morale/late-arrival signals require **≥2 independent** class-A/B sources before moving
   anything; a single source is logged, not applied.
3. GDELT used purely as a whitelisted-domain corroboration counter.
4. Hard effect caps + saturation; morale gets the **smallest cap**.
5. Fast decay on morale (half-life of days); injuries persist until a "returned to training"
   clearance.
6. **Provenance audit trail** per covariate: source URLs, source class, corroboration count,
   pre-cap raw value and post-cap applied value.
7. Whitelist > blacklist — only listed sources are readable.

---

## 9. Evaluation and validation protocol

- **Tuning:** Optuna Bayesian (TPE) search against a **rolling-origin (time-respecting)
  cross-validation** objective (OOS log-loss / RPS). Never random k-fold (it leaks future
  form). Nested so the test set is never seen during tuning.
- **Baselines:** v1, bookmaker market (de-vigged via penaltyblog), uniform.
- **Metrics:** log-loss, Brier, RPS + calibration / reliability curves (scoringrules /
  properscoring / sklearn / netcal).
- **Tournament replays:** WC2018, WC2022, Euro2024, Copa2024.
- **High-N club calibration tier:** thousands of league matches vs bookmaker closing odds
  (statistical power the sparse tournament replays lack).
- **Pre/post-LLM ablation:** identical model, context covariates zeroed vs active, scored
  out-of-sample — measures the context layer's marginal value rather than asserting it. This
  is the gate that decides whether morale leaves shadow mode.
- **WC2026 = sacred held-out test set**, scored exactly once at the end. Tuning never
  touches it.

---

## 10. Auto-tuning

- **Optuna** TPE sampler, seeded (`TPESampler(seed=...)`), RDB/journal storage so studies
  are **checkpointed and resumable** across an overnight run (the storage doubles as the run
  tracker — no MLflow/W&B needed).
- Objective: rolling-origin CV OOS log-loss/RPS using fast VI/Laplace fits per trial.
- Hyperparameter surface: `σ_att²`, `σ_def²`, prior strength `τ_prior`, competition-
  importance weights, context covariate caps and decay windows, and the hybrid-function
  hyperparameters (pyGAM smoothing / LightGBM params).
- CPU, all cores. The selected config is re-fit with full NUTS and frozen before the WC2026
  test scoring.

---

## 11. Tooling decisions (build vs leverage)

| Component | Decision | Tool |
|---|---|---|
| Inference engine for dynamic DC | adopt engine, build model | **NumPyro** (Apache-2.0) |
| Static-DC / bivariate-Poisson baselines, odds de-vigging, Elo/Massey/Pi | adopt | **penaltyblog** (MIT) |
| Hyperparameter tuning + run tracking | adopt | **Optuna** (MIT) |
| Scoring rules + calibration | adopt | **scoringrules/properscoring, sklearn, netcal** |
| Hybrid learned functions | adopt (CV-select) | **pyGAM** and/or **LightGBM** |
| FBref/Understat/ClubElo data | keep | **soccerdata** (already vendored) |
| Dynamic state-space DC model spec | build | — |
| 48-team bracket Monte Carlo | build | — |
| clubform subsystem + league adjustment | build | — |
| LLM context layer + LaTeX prose | build | — |
| Cache-first fetch clients + ledger | build/keep | — (stricter than any third-party client) |

Watch-outs carried into implementation: penaltyblog's DC is **static** (baseline only, never
the core); JAX needs deterministic key threading for reproducibility; do not adopt Gaussian-
filter libraries for the non-Gaussian Poisson emission; socceraction is frozen (citable, not
a hard dependency); no SaaS experiment trackers in a public reproducible repo.


---

## 12. Papers (docs/v2)

- **Methodology** — the dynamic state-space DC, hybrid learned prior/matchup, context layer,
  full math.
- **Validation** — backtests, calibration, the high-N club tier, convergence diagnostics,
  and the pre/post-LLM ablation.
- **Forecast** — the 2026 predictions (filled once WC2026 data arrives).
- **Comparison (v1 vs v2)** — the cheap-vs-powerful story with mathematical annotation and
  the ablation result.

Every modeling choice cites a reference in the shared bibliography (repo hard rule).

---

## 13. Operational sequencing

WC2026 actuals are **not** in the repo yet (the cache predates kickoff) and the bracket is
**not** confirmed. Therefore:

1. **Build now:** the full v2 scaffold, fetch layer (extended API-Football pulls), clubform,
   context layer, model, sim, tuner, evaluation harness, and the backtests on
   WC2018/WC2022/Euro2024/Copa2024 + the club tier — all runnable on cached/fetchable
   historical data.
2. **Validate now** against history; tune on rolling-origin CV.
3. **When 2026 data is available:** fetch actuals + confirmed bracket, run the pre/post-LLM
   ablation, score against the held-out WC2026 test set, and fill the forecast/comparison
   papers.

---

## 14. Acceptance criteria

- v2 **strictly beats v1** on log-loss, Brier, and RPS on the rolling out-of-sample
  internationals and on the tournament replays.
- v2 is **at least competitive with the de-vigged market** on the club calibration tier
  (v1 was not).
- Reliability curves show **no material miscalibration** after any isotonic/temperature
  correction.
- The pre/post-LLM ablation produces a **signed, significant** estimate of the context
  layer's value; morale leaves shadow mode **only** if that estimate is positive.
- The entire pipeline **reproduces exactly** from seeds on a fresh machine.

---

## 15. Risks and guardrails

- **Overfitting on sparse international data** → parsimonious structural core, few meaningful
  hyperparameters, strict time-series CV, nested test isolation.
- **Club→country transfer mis-set** → learned and validated, never assumed; ablate it.
- **LLM hype contamination** → structured-first, ≥2-source corroboration, smallest caps +
  fast decay on morale, shadow mode, full provenance audit.
- **Reproducibility drift from JAX** → deterministic key-threading harness; seed every
  stochastic step (repo hard rule).
- **Scope creep into premium event data** → gated behind a measured ablation trigger, not
  bought up front.
