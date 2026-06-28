PAPERS = 01-methodology 02-validation 03-forecast
V2_PAPERS = 01-methodology 02-comparison

.PHONY: docs docs-v2 test lint forecast all-data $(PAPERS) $(addprefix v2-,$(V2_PAPERS))

docs: $(PAPERS)

$(PAPERS):
	cd docs/v1/tex/$@ && tectonic -o ../../pdf $@.tex

docs-v2: $(addprefix v2-,$(V2_PAPERS))

$(addprefix v2-,$(V2_PAPERS)):
	cd docs/v2/tex/$(patsubst v2-%,%,$@) && tectonic -o ../../pdf $(patsubst v2-%,%,$@).tex

test:
	uv run pytest

lint:
	uv run ruff check src tests

# Every fetcher with per-step isolation and an end summary. Safe to run on
# any machine: blocked sources fail individually, everything else proceeds.
all-data:
	uv run python -m cupcast.v1.fetch.all

# Re-download refreshable caches (API-Football stays cache-first to protect
# the request ledger).
all-data-force:
	uv run python -m cupcast.v1.fetch.all --force

# Diagnose keys, per-source reachability (with redirect chains), cache state.
doctor:
	uv run python -m cupcast.v1.fetch.all --doctor

# Run-day: refresh results to the latest matches, snapshot odds (skipped if no
# key), regenerate every output and paper fragment, rebuild the PDFs.
forecast:
	uv run python -c 'from cupcast.v1.fetch.martj42 import fetch_results_csv; fetch_results_csv(refresh=True)'
	-uv run python -m cupcast.v1.fetch.odds
	uv run python -m cupcast.v1.report.build
	$(MAKE) docs

# v2 aggressive pull: international + club fixtures/lineups/players/injuries.
# Cache-first + ledgered; resumes from cache if a day's quota is exhausted.
v2-data:
	uv run python -m cupcast.v2.fetch.pull
