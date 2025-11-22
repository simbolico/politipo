# Repository Guidelines

## Project Structure & Modules
- `src/politipo.py`: core library module and public API.
- `src/test_politipo.py`: pytest suite (Hypothesis property tests included).
- `pyproject.toml`: build metadata and tool config (ruff, black, pytest, coverage).
- `Makefile`: common tasks for dev, CI, and releases.
- `.github/workflows/`: CI for lint/format/type/tests on Python 3.13/3.14.
- `dist/` and `build/`: generated artifacts. Do not edit by hand.

Keep new code in `src/`. Name tests `test_*.py` under `src/` to match pytest config.

## Build, Test & Development
- `make sync`: install dev deps via uv (`[dev]`).
- `make lint` / `make fix`: ruff check / autofix.
- `make fmt` / `make fmt-check`: black format / verify.
- `make type`: strict type check with `ty` (zero-warning policy).
- `make test`: run pytest quietly; `make cov` adds coverage report.
- `make extras-all`: install optional extras (`.[all]`) for Arrow/Polars/DuckDB/Pandera.
- `make build`: build sdist + wheel with uv. `make publish` uses Trusted Publishing.
- Python pinning: `make pin-3.13` (recommended) or `make pin-3.14`.

Examples:
```
uv pip install -e .           # editable install
PYTHONPATH=src uv run pytest  # run tests directly
```

## Coding Style & Naming
- Python 3.13+, line length 100, 4-space indent.
- Type hints required for public APIs; prefer precise, warning-free types.
- Tools: ruff (E,F,B,I,UP) and black. Run `make fix fmt` before commits.
- Naming: `snake_case` for functions/vars; `PascalCase` for classes; `UPPER_SNAKE` for constants; tests `test_*`.

## Testing Guidelines
- Frameworks: pytest + hypothesis. Coverage threshold: 97% (enforced via `pyproject.toml`).
- Location/pattern: tests in `src/` named `test_*.py`.
- Optional deps: guard with `pytest.mark.skipif` and install via `make extras-all` when needed.
- Run locally: `make test` or `make cov`. Add focused, property-based tests for new features.

## Commit & PR Guidelines
- Use Conventional Commits: `feat|fix|docs|chore|refactor|build|test(scope): message`.
  - Example: `feat(resolver): add Vector[N] support`.
- PRs: clear description, link issues, note optional-deps impact, and include examples or DDL output when relevant.
- Before submitting: `make lint fmt-check type test` must pass; update README if user-facing.

## Security & Configuration
- Minimal core deps; extras are opt-in. Avoid committing secrets; publishing uses Trusted Publishing.
- Makefile exports `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to smooth Python 3.14 builds—no action needed.
