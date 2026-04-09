.PHONY: help install install-dev test lint format build check-dist ci release-local

PYTHON ?= python3

help:
	@echo "Available commands:"
	@echo "  install       Install the package"
	@echo "  install-dev   Install package with dev and test dependencies"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run ruff"
	@echo "  format        Format code with ruff"
	@echo "  build         Build source and wheel distributions"
	@echo "  check-dist    Validate built distributions with twine"
	@echo "  ci            Run the local CI pipeline"
	@echo "  release-local Build and optionally upload to PyPI using env vars"

install:
	$(PYTHON) -m pip install .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,test]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

build:
	$(PYTHON) -m build --no-isolation

check-dist:
	$(PYTHON) -m twine check dist/*

ci:
	./scripts/run_ci_locally.sh

release-local:
	./scripts/release_to_pypi.sh

.DEFAULT_GOAL := help
