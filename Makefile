PAPERS = 01-methodology 02-validation 03-forecast

.PHONY: docs test lint forecast all-data $(PAPERS)

docs: $(PAPERS)

$(PAPERS):
	cd docs/tex/$@ && tectonic -o ../../pdf $@.tex

test:
	uv run pytest

lint:
	uv run ruff check src tests

# One-time data build on a fresh (unmanaged) machine.
all-data:
	uv run python -m cupcast.fetch.pull run --from 2022 --to 2024
	uv run python -c 'from cupcast.fetch.martj42 import fetch_results_csv; fetch_results_csv()'
	uv run python -m cupcast.fetch.squads
	uv run python -m cupcast.fetch.football_data
	uv run python -m cupcast.fetch.fbref

# Run-day: refresh results to the latest matches, snapshot odds (skipped if no
# key), regenerate every output and paper fragment, rebuild the PDFs.
forecast:
	uv run python -c 'from cupcast.fetch.martj42 import fetch_results_csv; fetch_results_csv(refresh=True)'
	-uv run python -m cupcast.fetch.odds
	uv run python -m cupcast.report.build
	$(MAKE) docs
