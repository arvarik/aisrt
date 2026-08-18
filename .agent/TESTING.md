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
- **Memory Bound Checks**: Exercise `MemoryBudget` directly. Confirm a waiter blocks until a release, that a file larger than the whole budget still runs rather than deadlocking, and that a release never drives the counter negative.

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

---

## Backend Route Coverage Matrix

_Populated by the SDET during the Trap phase. One row per CLI command or pipeline entry point._

| Entry Point / Command | Input Type | Valid Input | Invalid Input | Error Recovery | Edge Cases |
|-----------------------|------------|-------------|---------------|----------------|------------|
| `aisrt scan <media_dir>` | CLI — dry-run directory scan | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `aisrt run <media_dir>` | CLI — live pipeline execution | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `aisrt run --watch` | CLI — daemon mode (continuous loop) | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `aisrt run --translate` | CLI — foreign audio → English | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `Pipeline._producer()` | Async — discovery → extraction queue | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `Pipeline._extractor_worker()` | Async — FFmpeg stdout → numpy array | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `Pipeline._inference_worker()` | Async — Whisper transcription → SRT | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `DiscoveryEngine.scan()` | Async — NAS-safe file crawl | UNTESTED | UNTESTED | UNTESTED | UNTESTED |
| `AtomicWriter.write_srt()` | Sync — POSIX atomic `.srt` swap | UNTESTED | UNTESTED | UNTESTED | UNTESTED |

---

## Frontend Component State Matrix

N/A — Frontend topology is not active for this project.

---

## ML / AI Evaluation Thresholds

_Populated by the ML Engineer during the Build phase. Track transcription quality metrics._

| Metric | Target | Current | Method | Eval Set | Model Config | Last Run |
|--------|--------|---------|--------|----------|--------------|----------|
| Word Error Rate (WER) | ≤ 10% | — | `jiwer` against reference transcripts | Not configured | `large-v3` / `float16` | — |
| Character Error Rate (CER) | ≤ 5% | — | `jiwer` against reference transcripts | Not configured | `large-v3` / `float16` | — |
| Processing Throughput (x realtime) | ≥ 5x on CUDA | — | `PipelineStats.total_audio_duration_secs / elapsed` | N/A | `large-v3` / `float16` | — |
| Model Load Time | ≤ 30s | — | Wall clock from `STTWorker.initialize()` | N/A | `large-v3` / `float16` | — |
| VAD False Positive Rate | ≤ 2% | — | Manual review of silence-heavy media | Not configured | All configs | — |

### Eval / Holdout Boundary
- **eval_set**: No eval set configured yet
- **holdout_set**: No holdout set configured yet — HUMAN-ONLY when created

## Bugs Found (Fix Phase Queue)
1. _(None currently)_

---

## Regression Scenarios (Persistent)

| Scenario | Last Verified | Notes |
|----------|---------------|-------|
| Zero-disk extraction via FFmpeg | | |
| Atomic POSIX swap of `.srt` | | |
| Bounded memory queue limits | | |

### Backend Route Coverage

| Route | Method | Auth | Contract Test | Integration Test |
|-------|--------|------|---------------|------------------|
| _Fill in your routes_ | | | | |

### ML/AI Evaluation Thresholds

| Metric | Baseline | Target | Current | Status |
|--------|----------|--------|---------|--------|
| _Fill in your metrics_ | | | | |

