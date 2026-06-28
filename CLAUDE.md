# Cupcast

The project is "Cupcast" (capital C) in all prose — README, docs, paper titles. Code follows PEP-8: the package, paths, and pyproject name stay lowercase `cupcast`.

Never run `git commit` or `git push` — the user handles all git history themselves. Finish work, leave the tree dirty, and report what changed.

Monte Carlo forecasting engine for the 2026 FIFA World Cup. Dixon-Coles goal model fitted on recency-weighted international results, blended via a shrinkage prior with minutes-weighted squad-quality composites, simulated 50,000 times over the official 48-team bracket.

## Commands

- `uv sync` — install dependencies
- `uv run pytest` — run tests
- `uv run ruff check src tests` — lint
- `make docs` — build the three LaTeX papers (requires `tectonic`)
- `make doctor` — preflight: keys, per-source reachability with redirect chains, cache state
- `make all-data` — every fetcher with per-step isolation and an end summary
- `make all-data-force` — re-download refreshable caches (API-Football stays ledgered)
- `make forecast` — run day: refresh results, snapshot odds, 50k sims, all outputs, papers

## Layout

- `src/cupcast/fetch/` — cached API clients (API-Football, The Odds API, FBref)
- `src/cupcast/features/` — match table, squads, expected minutes, squad composites
- `src/cupcast/ratings/` — World Football Elo engine
- `src/cupcast/model/` — Dixon-Coles MLE, shrinkage prior, calibration
- `src/cupcast/sim/` — group stage, FIFA tiebreakers, R32 annex bracket, knockouts, Monte Carlo
- `src/cupcast/validate/` — club-data and tournament backtests, scoring metrics
- `src/cupcast/report/` — generates `outputs/` artifacts and `docs/tex/figures/`
- `docs/tex/` — LaTeX paper sources (numbered folders define reading order; `shared/` holds preamble + bibliography)
- `docs/pdf/` — committed PDF builds
- `outputs/` — committed final artifacts, split by model: `outputs/v1/` (baseline, via `cupcast.v1.report.build`) and `outputs/v2/` (via `cupcast.v2.report`, `make forecast-v2`), the latter segmented into `pretournament/` (held-out forecast) and `live/` (conditioned on the actual draw) plus `validation.md`

## Hard rules

- `data/` is never committed — it holds fetched/paid datasets. Only derived findings go in `docs/` and `outputs/`.
- API keys live in `.env` (gitignored); `.env.example` lists the required keys. Never print keys in logs or error messages.
- No Transfermarkt or any market-value data anywhere in the model. Performance metrics only.
- Every stochastic step takes an explicit seed. Results must reproduce exactly.
- Every modeling choice cites a reference in `docs/tex/shared/references.bib`.
- API clients are cache-first: every response is persisted under `data/raw/` on first fetch and the pipeline reads only from cache. API-Football calls go through the request ledger (`data/raw/api_football/ledger.json`) to respect the free-tier daily quota — never bypass it.
- This repo will be public. Nothing personal (paths, usernames, work references) in committed files.
