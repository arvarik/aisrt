# Architecture

_This document acts as the definitive anchor for understanding system design, data models, API contracts, and technology boundaries. Update this document during the Design and Review phases._

## 0. Project Topology

**Topology:** `[backend, ml-ai]`

_Agents: Read the corresponding Gemstack topology profiles (`backend.md` and `ml-ai.md`) from `~/.gemini/antigravity/global_workflows/` before proceeding with any workflow step. These profiles enforce data integrity testing, Evaluation-Driven Development (EDD), circuit breaker cost controls, and prompt versioning._

## 1. Tech Stack & Infrastructure
- **Language / Runtime**: Python 3.11+
- **Backend / API**: Typer CLI, no frontend
- **Database**: SQLite (WAL mode) via `aiosqlite`
- **Deployment**: Docker Compose or native Python environment (`pip install -e .` or `poetry`)
- **Package Management**: Poetry / pip (`pyproject.toml`)
- **Build System**: N/A (interpreted script)
- **Machine Learning**: `faster-whisper` (CTranslate2 backend) auto-routed based on hardware profiles (CUDA, Apple Silicon, CPU).

## 2. System Boundaries & Data Flow
### Pipeline Lifecyle Flow
1. **Discovery (`discovery.py`)**: Crawls NAS mounts safely, evaluating `inode` state, sibling SRT presence, and internal embedded subtitle tracks via `ffprobe` before queueing.
2. **Extraction (`extractor.py`)**: Spawns `ffmpeg` subprocess. Streams `-f s16le -` via `stdout` into an unbounded dynamically growing `bytearray` (zero-disk), finally converting to a monolithic `numpy.ndarray`.
3. **Inference (`stt.py`)**: Wraps `faster-whisper.BatchedInferencePipeline`. Submits the `numpy.ndarray` via a `ThreadPoolExecutor` to bypass GIL and prevent asyncio event loop starvation.
4. **Assembly (`assembly.py`)**: Parses the raw segment timestamps, applies punctuation/chunking heuristics for broadcast readability, and commits to NAS using `os.replace()` for POSIX atomic safety.

### Concurrency / Threading Model
- Strictly bounded `asyncio.Queue` bounds: extraction queue `maxsize=3`, inference queue `maxsize=2`. This prevents the CPU from flooding RAM with idle audio buffers while waiting for the GPU.
- Event Loop: `faster-whisper` inference and blocking operations are offloaded to `loop.run_in_executor()` using `STTWorker`'s dedicated thread pool. NEVER block the main `asyncio` event loop.

## 3. Data Models & Database Schema
- **State Tracker**: `aiosqlite` in WAL mode for NAS-safe file tracking.
- **Deduplication Key**: Composite key of `inode` + `size`. NEVER use `device_id` (volatile on network mounts).
- **State Flow**: State is immediately updated to `EXTRACTING` or `INFERENCING` the moment a file is popped off a queue. Do not wait for the pipeline to finish to update state.

## 4. API Contracts
- CLI tool interface defined in `src/aisrt/cli.py` using Typer.
- Commands: `run` (process media) and `scan` (dry-run profiling).

## 5. External Integrations / Hardware Acceleration
- **Intelligent Routing (`hardware.py`)**: Auto-detects system specs. 
  - VRAM >= 12GB: Routes to `large-v3` with `float16`.
  - VRAM >= 8GB: Routes to `large-v3-turbo`.
  - Edge cases fallback to `int8` quantization or pure CPU execution.
  - Apple Silicon routes directly to Metal Performance Shaders (MPS).
- **Media Processing**: Shells out to `ffmpeg`/`ffprobe` subprocesses. Avoids python AV bindings to minimize memory leaks and maximize codec compatibility.

### Model Ledger (ML/AI Topology)

_Documents every ML model in use. Required by the ml-ai topology profile for Circuit Breaker calculations._

| Model | Role | Resource Requirements | Context/Input Limits | Structured Output | Rate Limit | Circuit Breaker Cap |
|-------|------|----------------------|---------------------|-------------------|------------|---------------------|
| `large-v3` | Primary transcription (high-VRAM CUDA) | CUDA GPU, VRAM ≥ 10 GB, `float16` compute | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3-turbo` | Primary transcription (mid-VRAM CUDA) | CUDA GPU, VRAM ≥ 6 GB, `float16` compute | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3-turbo` | Primary transcription (low-VRAM CUDA) | CUDA GPU, VRAM ≥ 4 GB, `int8_float16` compute | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `large-v3-turbo` | Primary transcription (Apple Silicon / high-RAM CPU) | CPU, RAM > 16 GB, `int8` compute, all physical cores | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |
| `small.en` | Fallback transcription (low-resource CPU) | CPU, RAM ≤ 16 GB, `int8` compute, all physical cores | Unbounded audio input (streamed segments via VAD) | SRT subtitle segments with word-level timestamps | N/A (local inference) | N/A (local compute) |

_Model selection is controlled by `ModelRouter.get_config()` in `src/aisrt/hardware.py`. User overrides via `--force-model`, `--force-device`, and `AISRT_HARDWARE__FORCE_MODEL` env var._


## 6. Invariants & Safety Rules
- **Zero-Disk Extraction**: NEVER write `.wav`, `.mp4`, or `.ts` temp files to disk.
- **Asynchronous Memory Safety**: NEVER uncap the `asyncio.Queue` bounds. ALWAYS explicitly delete `job.audio_data` and invoke `gc.collect()` after inference.
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
- **Install**: `poetry install` or `pip install -e ".[dev]"`
- **Start Dev Mode**: `python -m src.aisrt.cli run /path/to/media --dry-run` or `aisrt run ...`
- **Database Setup**: SQLite file auto-created on first run.

## 10. Environment Variables
- Handled via CLI flags or `pydantic-settings` environment variables in Docker (e.g., `HF_TOKEN`, `AISRT_TRANSLATE`, `AISRT_WATCH`, `AISRT_HARDWARE__FORCE_MODEL`).
