# Cupcast I — baseline outputs

The frozen v1 model (static, recency-weighted Dixon–Coles with an Elo-and-squad
shrinkage prior) — the "cheap/simple" baseline that v2 is measured against.

Regenerate with:

```
make forecast        # refresh results, snapshot odds, rebuild v1 outputs + papers
# or just the report:
uv run python -m cupcast.v1.report.build
```

This writes `simulation_results.csv`, `top3_forecast.md`, `executive_summary.md`,
`market_comparison.{md,csv}`, `sensitivity.md`, `match_predictions*.{md,csv}`, and
`validation.md` here. The v2 forecast lives in `../v2/`; the head-to-head
comparison is in `docs/v2/pdf/02-comparison.pdf`.
