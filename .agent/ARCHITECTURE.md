# Architecture

_This document acts as the definitive anchor for understanding system design, data models, API contracts, and technology boundaries. Update this document during the Design and Review phases._

## 1. Tech Stack & Infrastructure
- **Language / Runtime**: Python 3.11+
- **Backend / API**: Typer CLI, no frontend
- **Database**: SQLite (WAL mode) via `aiosqlite`
- **Deployment**: Docker Compose or native Python environment (`pip install -e .` or `poetry`)
- **Package Management**: Poetry / pip (`pyproject.toml`)
- **Build System**: N/A (interpreted script)
- **Machine Learning**: `faster-whisper` (CTranslate2 backend) auto-routed based on hardware profiles (CUDA, Apple Silicon, CPU).

## 2. System Boundaries & Data Flow
### Request / Data Flow
- **Pipeline**: CLI entrypoint → `asyncio.TaskGroup` producer-consumer orchestrator → bounded `asyncio.Queue` → GPU inference worker in `ThreadPoolExecutor(max_workers=1)` → atomic POSIX file writer.
- **Audio Extraction**: `ffmpeg` streaming via `stdout` (`-f s16le -`) dynamically into a `bytearray` and converted to `numpy.ndarray`. 

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

## 5. External Integrations / AI
- Integrates `faster-whisper` with intelligent hardware routing (auto-detects VRAM and routes to `large-v3`, `large-v3-turbo`, `small`, or `int8` models).
- Shells out to `ffmpeg`/`ffprobe` for media analysis and zero-disk streaming.

## 6. Invariants & Safety Rules
- **Zero-Disk Extraction**: NEVER write `.wav`, `.mp4`, or `.ts` temp files to disk.
- **Asynchronous Memory Safety**: NEVER uncap the `asyncio.Queue` bounds. ALWAYS explicitly delete `job.audio_data` and invoke `gc.collect()` after inference.
- **Event Loop Starvation**: NEVER execute `faster-whisper` inference inside the main `asyncio` event loop.
- **NAS-Safe Database State**: NEVER map the DB to an NFS/SMB network share. NEVER use `device_id` for deduplication. IMMEDIATELY update state upon queue pop.
- **POSIX Atomic Subtitle Writing**: NEVER write directly to `.srt`. Always write to `.movie.srt.tmp` first. ALWAYS inherit `st_uid`, `st_gid`, and `st_mode` from the source MKV using `os.chown` and `os.chmod` on the temp file. ALWAYS finalize with a true POSIX atomic swap: `os.replace()`.
- **Architectural Boundary Changes**: If you change an architectural boundary, you MUST explicitly document the justification.

## 7. Error Handling Patterns
- Memory streams must not be blocked by arbitrary `.communicate()` limits.
- Bounded concurrency protects against OOM (Out-of-Memory) errors during 24/7 watch loops.

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
