# Cupcast

Monte Carlo forecasting engine for the 2026 FIFA World Cup.

I fit a Dixon-Coles goal model on a decade of international results (recency-weighted, opponent-adjusted), blend the fitted team ratings with squad-quality composites built from minutes-weighted on-pitch performance data, and simulate the full 48-team tournament 50,000 times — group tiebreakers, the official Round-of-32 bracket allocation, extra time and penalty shootouts included.

## Forecast

Headline probabilities land here once the pipeline is complete. Full analysis lives in the papers below.

## Documentation

Three papers under [`docs/pdf/`](docs/pdf/):

1. **Methodology** — data sources, Elo ratings, the goal model, squad composites, simulation design
2. **Validation** — backtests on club football and Euro 2024 / Copa América 2024, scoring rules, calibration
3. **Forecast** — the 2026 predictions, sensitivity analysis, comparison against bookmaker odds

Sources are in [`docs/tex/`](docs/tex/); build with `make docs` (requires [tectonic](https://tectonic-typesetting.github.io/)).

## Running it

```sh
uv sync
cp .env.example .env   # add your API keys
uv run pytest
```

### Rebuilding the data on a fresh machine

The `data/` directory is never committed (licensed/fetched datasets). Either copy it
from an existing machine, or rebuild it:

```sh
make doctor          # preflight: keys, per-source reachability, cache state
make all-data        # every fetcher, per-step isolation, end summary
make all-data-force  # re-download refreshable caches (API-Football stays ledgered)
```

Required sources: API-Football (key, ledgered), the martj42 results dataset,
Wikipedia squads, and Understat (player npxG+xA — feeds the squad composite).
Optional: football-data.co.uk (club closing odds for the validation tier),
The Odds API (key; market comparison), and FBref (goalkeeper data for the
shootout adjustment; drives a real browser — install Chrome/Chromium first).
Filtered corporate networks block some sources; `make doctor` shows exactly
which, with redirect chains. Building the papers needs `tectonic` (in the
Arch repos and Homebrew). All fetchers are cache-first: rerunning never
refetches what exists.

### The forecast

```sh
make all-data   # one-time on a fresh machine (see network notes above)
make forecast   # refresh results, snapshot odds, 50k sims, all outputs, papers
```

`make forecast` regenerates everything in `outputs/` (simulation results, all
104 match predictions, top-3 forecast, sensitivity, validation, executive
summary, market comparison) and rebuilds the three PDFs with the new numbers.
Everything is CPU-bound numpy/scipy — no GPU involved — and the full pipeline
runs in well under a minute once data is cached.

## License

MIT
