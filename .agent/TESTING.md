# Testing Strategy & Results

_This file tracks test methods, scenarios, and results with concrete execution evidence. Bugs found here block the release of a feature. Agents must update this during the Test and Fix phases._

## 0. Local Development Setup
### Prerequisites
- Python 3.11+
- `ffmpeg` installed on the host system.
- NVIDIA GPU (optional, for CUDA acceleration).

### Start the App
- `aisrt run /path/to/media`
- `aisrt scan /mnt/movies --min-age-mins 60` (Dry Run)

### Database
- SQLite state DB auto-created on first run.

## 1. Test Methods & Tools
### Unit / Integration Tests
- **Run all tests**: `pytest tests/` (using `pytest-asyncio` for async tests).
- **Async Fixtures**: All core worker logic tests must utilize `pytest.mark.asyncio` and yield safely closed `asyncio.Queue` objects.
- **NAS-Edge Mocking**: Extensively mock `os.stat` to simulate NAS environment states (e.g., changing `inode` across network dropouts, or testing `os.replace` throwing `EACCES` permission errors).
- **OOM Constraint Checks**: Inject mock oversized `bytearray` streams to validate that `asyncio.Queue(maxsize=3)` successfully blocks the extraction producer, verifying the daemon will not crash the host's RAM.

### Type Checking & Linting
- **Type Checking**: `mypy src/aisrt tests --strict` (must produce 0 errors).
- **Linting**: `ruff check .`
- **Formatting**: `ruff format --check .`

## 2. Execution Evidence Rules
- For Python tests, paste the output of `pytest` showing PASS/FAIL lines.
- For type checking / linting, paste the command and its output (e.g., `mypy src/aisrt tests --strict → 0 errors`).
- "PASS" with no evidence is treated as UNTESTED.

---

## Current Feature Scenarios: Project Bootstrapping

| Scenario | Status | Notes (Evidence) |
|----------|--------|------------------|
| Empty/null/missing inputs | UNTESTED | |
| Valid payload creates resource | UNTESTED | |
| Invalid payload returns structured error | UNTESTED | |

## Bugs Found (Fix Phase Queue)
1. _(None currently)_

---

## Regression Scenarios (Persistent)

| Scenario | Last Verified | Notes |
|----------|---------------|-------|
| Zero-disk extraction via FFmpeg | | |
| Atomic POSIX swap of `.srt` | | |
| Bounded memory queue limits | | |
