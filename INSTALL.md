# Installation

## Requirements

- Python 3.10 or newer
- `pip`

## Install From Source

```bash
git clone https://github.com/HenriquesLab/mAIcroscopysandbox.git
cd mAIcroscopysandbox
python -m pip install -e .
```

## Development Install

Install the package together with linting and testing tools:

```bash
python -m pip install -e ".[dev,test]"
```

## Local Validation

Run the same checks used by the repository automation:

```bash
./scripts/run_ci_locally.sh
```

That command runs:

1. `ruff check`
2. `pytest` with coverage
3. `python -m build --no-isolation`
4. `twine check`

