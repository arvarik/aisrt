# Style Guide & Code Conventions

_This document enforces the visual identity and coding patterns of the project. It prevents context drift as multiple agents work on the codebase. Agents MUST follow these rules strictly._

## 1. Visual Language & Tokens
- **Terminal UI (`Rich`)**: Output formatting is handled exclusively via `Rich`. NEVER use standard `print()`.
- **Progress Bars**: Use `rich.progress` to represent FFmpeg extraction metrics and CTranslate inference batch progress. Ensure bars gracefully decay if `stdout` is piped to a file.
- **Logging**: Loguru writes to stderr. Rich owns stdout. Never drive a Rich live display from a worker thread; post the update to the event loop with `loop.call_soon_threadsafe` and let the main thread render it.

## 2. CLI Component Patterns
- **Typer Arguments**: CLI arguments (`aisrt run`) MUST be defined via `typer.Option`. Use exact `kebab-case` for flags (e.g. `--min-age-mins`).
- **Pydantic Validation**: All environment variables and configurations (`config.py`) must be strictly parsed via `pydantic-settings`. Do not use raw `os.environ.get()` inside business logic.

## 3. Code Conventions
### Architecture Patterns
- **Concurrency**: `asyncio.TaskGroup` orchestrator handles concurrent pipelines. Blocking workloads (like `faster-whisper` inference or heavy numpy transpositions) MUST be offloaded to `loop.run_in_executor()`.
- **No Temporary Files**: Zero-disk extraction. FFmpeg streams `stdout` straight into a preallocated `numpy` float32 buffer. Do not stage the bytes in a `bytearray` first.

### State Management
- **Immediate State Tracking**: State in the SQLite DB is updated to `EXTRACTING` or `INFERENCING` immediately upon popping an item from a queue. No waiting for the full pipeline run.
- **Memory Management**: Drop the last reference to a large array as soon as the work is done. CPython frees it by reference counting immediately. Do NOT call `gc.collect()` per file: it only walks reference cycles, costs time proportional to the live heap, and stalls the event loop. Reserve and release bytes through `MemoryBudget` instead.

### Strict Typing
- All code MUST pass `uv run mypy`, which is configured for strict mode over `src/aisrt` and `tests`.
- No `Any` types unless absolutely necessary.
- Comprehensive type hints are required everywhere.
- All code MUST be formatted and linted via `ruff format .` and `ruff check .`.

## 4. Naming Conventions
- **Files**: `snake_case.py`
- **Variables / Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

## 5. Import Ordering
- Standard Python ordering enforced by `ruff`:
  1. Standard library (`os`, `asyncio`, etc.)
  2. Third-party (`pydantic`, `faster_whisper`, `rich`, etc.)
  3. Local modules (`from .extractor import AudioExtractor`)

## 6. Documentation Standards
- **Docstrings**: Use Google-style docstrings for all classes and methods. Follow existing idioms strictly.
- **README**: Keep README focused on setup and usage. Architecture details go in `.agent/ARCHITECTURE.md`.

## 7. Anti-Patterns (FORBIDDEN)
- ❌ NEVER write `.wav`, `.mp4`, or `.ts` temp files to disk.
- ❌ NEVER uncap the `asyncio.Queue` bounds.
- ❌ NEVER execute `faster-whisper` inference or blocking CTranslate2 operations inside the main `asyncio` event loop.
- ❌ NEVER map the DB to an NFS/SMB network share.
- ❌ NEVER use `device_id` as a deduplication key (volatile on network mounts).
- ❌ NEVER write directly to `.srt`. Always write to `.movie.srt.tmp` first.
- ❌ NEVER finalize subtitle creation without a true POSIX atomic swap: `os.replace(temp_srt_path, final_srt_path)`.
- ❌ NEVER use `Any` in type hints unless absolutely necessary.
- ❌ NEVER bypass the type system or disable `ruff` or `mypy` checks.
