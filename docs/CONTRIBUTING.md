# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,viz,notebook]'
```

## Quality Gates

- Run tests: `pytest`
- Optional lint/format check: `ruff check .`

## Deterministic vs Parallel

- Use `execution_mode="single"` for deterministic debugging and reproducibility checks.
- Use `execution_mode="parallel"` for faster evaluation on larger populations.
- Keep in mind that deterministic guarantees are strongest in single-process mode.

## Checklist for Behavior Changes

1. Add/adjust tests under `tests/`.
2. Keep checkpoint schema versioned.
3. Update migration notes if behavior intentionally diverges from JS logic.
