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
uv run python -m cupcast.fetch.pull run --from 2022 --to 2024  # API-Football (~30 requests, ledgered)
uv run python -c 'from cupcast.fetch.martj42 import fetch_results_csv; fetch_results_csv()'
uv run python -m cupcast.fetch.squads          # confirmed 26-man squads
uv run python -m cupcast.fetch.football_data   # club results + closing odds
uv run python -m cupcast.fetch.fbref           # 2025-26 player metrics (slow; drives a browser)
```

The last two fail on networks that filter football-data.co.uk or kill unsigned
browser drivers (typical corporate setups) — run them from an unmanaged machine.
The FBref pull drives a real browser: install Chromium first (`pacman -S chromium`
on Arch, `brew install --cask chromium` on macOS). Building the papers needs
`tectonic` (in the Arch repos and Homebrew). All fetchers are cache-first:
rerunning never refetches what exists.

### The forecast

```sh
uv run python -m cupcast.fetch.odds   # market snapshot (needs ODDS_API_KEY)
uv run python -m cupcast.run          # fit -> shrink -> 50k simulations -> outputs/
```

Everything is CPU-bound numpy/scipy — no GPU involved — and the full pipeline
runs in well under a minute once data is cached.

## License

MIT
