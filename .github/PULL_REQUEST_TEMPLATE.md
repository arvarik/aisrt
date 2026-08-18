## Summary

<!-- What does this change do, and why does it matter? One or two sentences. -->

## Changes

<!-- Bullet the user-visible behavior first, then the implementation. -->

-

## Testing

<!-- List the exact commands you ran and the result. -->

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/aisrt tests
uv run pytest
```

## Checklist

- [ ] The code passes `ruff check`, `ruff format --check`, `mypy`, and `pytest`.
- [ ] New behavior has a test.
- [ ] The documentation matches the code.
- [ ] `CHANGELOG.md` records the change under `Unreleased`.
