# Contributing to aisrt

Thank you for considering a contribution. This document tells you how to set the project
up, what the quality bar is, and how to get a change merged.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[hatchling](https://hatch.pypa.io/) for builds.

1. Install Python 3.11 or later, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install FFmpeg. The integration tests run the real binary.
   - Debian or Ubuntu: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
3. Clone and install:
   ```bash
   git clone https://github.com/arvarik/aisrt.git
   cd aisrt
   uv sync
   ```

`uv sync` creates the virtual environment and installs the project together with its
development group. Run any command through `uv run`, which activates that environment for
you.

## Quality checks

Every pull request must pass all four checks. Continuous integration runs the same
commands, so run them locally first.

```bash
uv run ruff check .        # Lint
uv run ruff format --check .  # Formatting
uv run mypy                # Strict type check
uv run pytest              # Tests, with an 85 percent coverage floor
```

`uv run ruff format .` rewrites the files in place when the format check fails.

### What the checks enforce

- **Ruff** runs pycodestyle, pyflakes, isort, bugbear, pyupgrade, bandit, pydocstyle, and
  flake8-async. Every public function needs a Google-style docstring.
- **Mypy** runs in strict mode over `src/aisrt` and `tests`. New code must be fully typed.
- **Pytest** measures coverage and fails below 85 percent.

## Tests

Unit tests mock the boundary, never the logic under test. Integration tests carry the
`integration` marker and run the real FFmpeg binary; they skip themselves when FFmpeg is
absent.

```bash
uv run pytest -m integration      # Only the integration tests
uv run pytest -m "not integration"  # Skip them
uv run pytest tests/test_assembly.py -v  # One module
```

Two rules keep the suite useful:

1. A test must fail if the code it covers is deleted. A test that only asserts a mock was
   called proves nothing.
2. Documentation claims are tested. `tests/test_docs.py` reads the README and checks every
   documented option and environment variable against the running program. If you add an
   option, document it and the test will confirm it works.

## Pull request process

1. Fork the repository and branch from `main`.
2. Make the change, with tests.
3. Run the four quality checks.
4. Record the change in `CHANGELOG.md` under `Unreleased`.
5. Open a pull request describing the problem and the fix.

## Reporting a security problem

Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
