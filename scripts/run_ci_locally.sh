#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e ".[dev,test]"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg}"

python3 -m ruff check .
python3 -m pytest
python3 -m build --no-isolation
python3 -m twine check dist/*
