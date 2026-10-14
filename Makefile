.PHONY: install test cov lint format suites mock-demo diagrams report clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

cov:
	$(PYTHON) -m pytest -q --cov --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

# Regenerate the committed suite snapshots from the generators.
suites:
	$(PYTHON) -m refusalscope generate --out suites

# End-to-end run on in-process MOCK servers. Output is labelled MOCK and git-ignored.
mock-demo:
	$(PYTHON) -m refusalscope mock-demo --out tmp/mock-demo

# Re-render docs/diagrams/*.mmd to SVG (needs node; see scripts/render_diagrams.sh).
diagrams:
	./scripts/render_diagrams.sh

# Tables and figures from results/*.json (refuses when there are none).
report:
	$(PYTHON) scripts/make_report.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov tmp report runs src/*.egg-info
