.PHONY: venv install precommit fmt lint test run cpu gpu clean

PY=python3
VENV=.venv
ACT=source $(VENV)/bin/activate

venv:
	$(PY) -m venv $(VENV)
	$(ACT); pip install -U pip wheel

install: venv
	$(ACT); pip install -e .; pip install -U pre-commit

precommit: install
	$(ACT); pre-commit install

fmt:
	$(ACT); black src tests; isort src tests

lint:
	$(ACT); ruff check src tests

test:
	$(ACT); PYTHONPATH=src pytest -q

run: cpu

cpu:
	$(ACT); bash scripts/run_local.sh

gpu:
	$(ACT); bash scripts/run_remote.sh

clean:
	rm -rf .venv dist build **/*.egg-info data/artifacts data/logs
