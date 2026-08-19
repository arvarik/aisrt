# Architecture

_This document acts as the definitive anchor for understanding system design, data models, API contracts, and technology boundaries. Update this document during the Design and Review phases._

## 0. Project Topology

**Topology:** `[backend, ml-ai]`

_Agents: Read the corresponding Gemstack topology profiles (`backend.md` and `ml-ai.md`) from `~/.gemini/antigravity/global_workflows/` before proceeding with any workflow step. These profiles enforce data integrity testing, Evaluation-Driven Development (EDD), circuit breaker cost controls, and prompt versioning._

## 1. Tech Stack & Infrastructure
- **Language / Runtime**: Python 3.11+
- **Backend / API**: Typer CLI, no frontend
- **Database**: SQLite (WAL mode) via `aiosqlite`
- **Deployment**: Docker Compose or a native Python environment
- **Package Management**: uv (`pyproject.toml`, `uv.lock`). There is no Poetry configuration.
- **Build System**: hatchling with hatch-vcs. The version comes from the git tags.
- **Machine Learning**: `faster-whisper` (CTranslate2 backend) auto-routed based on hardware profiles (CUDA, Apple Silicon, CPU).

## 2. System Boundaries & Data Flow
### Pipeline Lifecyle Flow
1. **Discovery (`discovery.py`)**: Crawls NAS mounts safely, evaluating `inode` state, sibling SRT presence, and internal embedded subtitle tracks via `ffprobe` before queueing.
2. **Extraction (`extractor.py`)**: Spawns an `ffmpeg` subprocess and streams `-f s16le -` from `stdout` directly into a preallocated `numpy` float32 buffer sized from the probed duration. Never accumulate into a `bytearray` and then call `.astype()`: that holds both buffers at once and raises peak memory by about 54 percent.
3. **Inference (`stt.py`)**: Owns one `WhisperModel`, or a `BatchedInferencePipeline` when a batch size is set. Submits the array through a dedicated single-worker `ThreadPoolExecutor` so the event loop never blocks.
4. **Assembly (`assembly.py`)**: Converts word timestamps into standards-compliant cues and commits them with `fsync` plus `os.replace()`.

### Concurrency / Threading Model
- Bounded queues plus a byte-denominated `MemoryBudget`. Queue depth alone is not a memory bound, because a three-hour film and a three-minute clip occupy the same slot. The budget defaults to 2048 MB and is set with `--max-memory-mb`.
- Event Loop: `faster-whisper` inference and blocking operations are offloaded to `loop.run_in_executor()` using `STTWorker`'s dedicated thread pool. NEVER block the main `asyncio` event loop.

## 3. Data Models & Database Schema
- **State Tracker**: `aiosqlite` in WAL mode for NAS-safe file tracking.
- **Deduplication Key**: Composite key of `device` + `inode` + `size`. The device number is required: two volumes can reuse an inode number without sharing content, and matching on inode alone wrongly skips files.
- **State Flow**: State moves to `EXTRACTING` or `INFERENCING` the moment a file is popped off a queue, so an interrupted run can be recovered. `reset_stale_states()` returns those rows to `PENDING` at startup.
- **Statuses**: `PENDING`, `EXTRACTING`, `INFERENCING`, `COMPLETED`, `NO_SPEECH`, `EMBEDDED_EXISTS`, `FAILED`.
- **Migrations**: `PRAGMA user_version` plus `ALTER TABLE`. NEVER drop the table: that destroys the user's history.

## 4. API Contracts
- CLI tool interface defined in `src/aisrt/cli.py` using Typer.
- Commands: `run` (process media) and `scan` (dry-run profiling).

## 5. External Integrations / Hardware Acceleration
- **Intelligent Routing (`hardware.py`)**: Reads the hardware and the task, then resolves a model, a device, and a precision. The table lives in `ModelRouter.get_config` and is mirrored in `README.md`.
  - VRAM >= 10 GB: `large-v3` with `float16`.
  - VRAM >= 8 GB: `large-v3` with `int8_float16`.
  - VRAM >= 6 GB: `large-v3-turbo` with `float16`.
  - VRAM >= 4 GB: `large-v3-turbo` with `int8_float16`.
  - No CUDA, RAM >= 16 GB: `large-v3-turbo` on the CPU with `int8`.
  - Anything smaller: `small.en`.
  - **Apple Silicon runs on the CPU.** CTranslate2 has no Metal backend, so there is no MPS path. `int8` is the fastest option there because it is the only multi-threaded one.
  - **Translation changes the model.** The turbo checkpoint and every `*.en` checkpoint return the original language when asked to translate, so `--translate` routes to `large-v3` or `medium`.
- **Media Processing**: Shells out to `ffmpeg`/`ffprobe` subprocesses. Avoids python AV bindings to minimize memory leaks and maximize codec compatibility.

### Model Ledger (ML/AI Topology)

_Documents every ML model in use. Required by the ml-ai topology profile for Circuit Breaker calculations._

| Model | Role | Resource Requirements | Context/Input Limits | Structured Output | Rate Limit | Circuit Breaker Cap |
|-------|------|----------------------|---------------------|-------------------|------------|---------------------|
| `large-v3` | Primary transcription (high-VRAM CUDA) | CUDA GPU, VRAM ≥ 10 GB, `float16` compute | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3` | Primary transcription (8-10 GB CUDA) | CUDA GPU, VRAM ≥ 8 GB, `int8_float16` compute | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3-turbo` | Primary transcription (4-6 GB CUDA) | CUDA GPU, VRAM ≥ 4 GB, `int8_float16` compute. `medium` replaces it when translating. | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3-turbo` | Primary transcription (Apple Silicon or high-RAM CPU) | CPU, RAM ≥ 16 GB, `int8` compute, all physical cores | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `small.en` | Fallback transcription (low-resource) | CPU, RAM < 16 GB, `int8` compute, all physical cores. `small` replaces it when translating. | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |

_Model selection is controlled by `ModelRouter.get_config()` in `src/aisrt/hardware.py`. User overrides via `--force-model`, `--force-device`, and `AISRT_HARDWARE__FORCE_MODEL` env var._


## 6. Invariants & Safety Rules
- **Zero-Disk Extraction**: NEVER write `.wav`, `.mp4`, or `.ts` temp files to disk.
- **Asynchronous Memory Safety**: NEVER uncap the `asyncio.Queue` bounds, and always reserve from `MemoryBudget` before decoding. Queue depth bounds the file count; the budget bounds the bytes, which is what actually protects the host.
- **Event Loop Starvation**: NEVER execute `faster-whisper` inference inside the main `asyncio` event loop.
- **NAS-Safe Database State**: NEVER map the DB to an NFS/SMB network share. NEVER use `device_id` for deduplication. IMMEDIATELY update state upon queue pop.
- **POSIX Atomic Subtitle Writing**: NEVER write directly to `.srt`. Always write to `.movie.srt.tmp` first. ALWAYS inherit `st_uid`, `st_gid`, and `st_mode` from the source MKV using `os.chown` and `os.chmod` on the temp file. ALWAYS finalize with a true POSIX atomic swap: `os.replace()`.
- **Architectural Boundary Changes**: If you change an architectural boundary, you MUST explicitly document the justification.

## 7. Error Handling Boundaries
- **FFmpeg Subprocess Timeouts**: Must use `asyncio.wait_for()` to prevent zombie `ffmpeg` process holding up queues indefinitely if network drops.
- **Memory Segmentation Faults**: If CTranslate2 crashes randomly, the worker must gracefully recover and un-queue the file rather than halting the daemon.
- **Database Locks**: Handle `sqlite3.IntegrityError` or locked states gracefully. Retries should use exponential backoff, but given WAL mode, concurrent reads shouldn't block.
- **EACCES/EPERM Permissions**: Gracefully warn and skip processing if the `.movie.srt.tmp` cannot inherit POSIX permissions of the source `.mkv`.

## 8. Directory Structure
- `src/aisrt/cli.py`: Typer CLI entrypoint.
- `src/aisrt/pipeline.py`: The `asyncio.TaskGroup` producer-consumer orchestrator.
- `src/aisrt/discovery.py`: NAS-safe file crawler.
- `src/aisrt/extractor.py`: Zero-disk FFmpeg stdout stream processor.
- `src/aisrt/hardware.py`: Profiler for CUDA, Apple Silicon, and CPU. Routes to optimal Whisper model.
- `src/aisrt/stt.py`: Singleton worker encapsulating the Whisper model and `BatchedInferencePipeline`.
- `src/aisrt/assembly.py`: Broadcast-quality SRT chunker and Atomic POSIX writer.
- `src/aisrt/state.py`: SQLite async state tracker.

## 9. Local Development
- **Install**: `uv sync`
- **Start Dev Mode**: `python -m src.aisrt.cli run /path/to/media --dry-run` or `aisrt run ...`
- **Database Setup**: SQLite file auto-created on first run.

## 10. Environment Variables
- Handled via CLI flags or `pydantic-settings` environment variables in Docker (e.g., `HF_TOKEN`, `AISRT_TRANSLATE`, `AISRT_WATCH`, `AISRT_HARDWARE__FORCE_MODEL`).
