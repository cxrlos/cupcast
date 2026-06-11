PAPERS = 01-methodology 02-validation 03-forecast

.PHONY: docs test lint $(PAPERS)

docs: $(PAPERS)

$(PAPERS):
	cd docs/tex/$@ && tectonic -o ../../pdf $@.tex

test:
	uv run pytest

lint:
	uv run ruff check src tests
