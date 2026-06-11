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

Pipeline commands will be documented here as the stages land. The underlying datasets are fetched via APIs (API-Football, The Odds API, FBref) and are not redistributed in this repo — the fetch scripts rebuild them from your own keys.

## License

MIT
