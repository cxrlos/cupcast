PAPERS = 01-methodology 02-validation 03-forecast

.PHONY: docs test lint forecast all-data $(PAPERS)

docs: $(PAPERS)

$(PAPERS):
	cd docs/tex/$@ && tectonic -o ../../pdf $@.tex

test:
	uv run pytest

lint:
	uv run ruff check src tests

# Every fetcher with per-step isolation and an end summary. Safe to run on
# any machine: blocked sources fail individually, everything else proceeds.
all-data:
	uv run python -m cupcast.fetch.all

# Re-download refreshable caches (API-Football stays cache-first to protect
# the request ledger).
all-data-force:
	uv run python -m cupcast.fetch.all --force

# Diagnose keys, per-source reachability (with redirect chains), cache state.
doctor:
	uv run python -m cupcast.fetch.all --doctor

# Run-day: refresh results to the latest matches, snapshot odds (skipped if no
# key), regenerate every output and paper fragment, rebuild the PDFs.
forecast:
	uv run python -c 'from cupcast.fetch.martj42 import fetch_results_csv; fetch_results_csv(refresh=True)'
	-uv run python -m cupcast.fetch.odds
	uv run python -m cupcast.report.build
	$(MAKE) docs
