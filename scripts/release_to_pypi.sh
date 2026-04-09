#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e ".[dev,test]"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg}"

python3 -m pytest
python3 -m build --no-isolation
python3 -m twine check dist/*

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "PYPI_TOKEN is not set; skipping upload after build and validation."
  exit 0
fi

REPOSITORY_URL="${PYPI_REPOSITORY_URL:-https://upload.pypi.org/legacy/}"
python3 -m twine upload --non-interactive \
  --repository-url "${REPOSITORY_URL}" \
  -u __token__ \
  -p "${PYPI_TOKEN}" \
  dist/*
