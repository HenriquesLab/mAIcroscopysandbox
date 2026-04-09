# Contributing

## Workflow

1. Create a feature branch.
2. Make your changes in small, reviewable commits.
3. Run the local checks before opening a pull request.
4. Open a PR with a short summary and any relevant benchmark or test notes.

## Local Checks

```bash
./scripts/run_ci_locally.sh
```

That is the preferred pre-push check because it matches CI.

## Code Style

- Keep docstrings concise and structured.
- Prefer Google-style docstrings for new public APIs.
- Use `ruff format` and `ruff check` to keep formatting and linting consistent.

## Testing Expectations

- Add tests for new public behavior.
- Keep overall coverage above 70 percent.
- Prefer deterministic tests with explicit seeds when randomness is involved.

## Release Process

- CI runs on `.github/workflows/ci.yml`
- Release publishing is handled by `.github/workflows/publish.yml`
- Local release validation is available via:

```bash
./scripts/release_to_pypi.sh
```

If you want to upload from a local environment, set `PYPI_TOKEN` first.

