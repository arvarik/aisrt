# 🤖 Ultimate AI-SRT Development Rules

You are an expert AI software engineer contributing to the **aisrt** project. This is a highly concurrent, zero-disk, hardware-aware pipeline for mass-generating broadcast-quality subtitles on NAS systems. 

Your code must be **Open Source Enterprise Grade**.

## 🛑 STRICT ARCHITECTURAL BOUNDARIES (NEVER VIOLATE)

1. **Zero-Disk Extraction (No SSD Wear):**
   - NEVER write `.wav`, `.mp4`, or `.ts` temp files to disk.
   - ALWAYS stream `ffmpeg` via `stdout` (`-f s16le -`) dynamically into a `bytearray` and convert to `numpy.ndarray`.
   - If you modify `AudioExtractor`, ensure memory streams are not blocked by arbitrary `.communicate()` limits.

2. **Asynchronous Memory Safety (OOM Protection):**
   - NEVER uncap the `asyncio.Queue` bounds.
   - Extraction queue `maxsize` is strictly capped at `3` to prevent the CPU from flooding RAM with idle audio buffers while waiting for the GPU.
   - Inference queue `maxsize` is strictly capped at `2`.
   - ALWAYS explicitly delete `job.audio_data` and invoke `gc.collect()` after inference to prevent memory fragmentation during 24/7 watch loops.

3. **Event Loop Starvation (GIL Deadlocks):**
   - NEVER execute `faster-whisper` inference or blocking CTranslate2 operations inside the main `asyncio` event loop.
   - ALWAYS offload to `loop.run_in_executor()` using the `STTWorker`'s dedicated `ThreadPoolExecutor(max_workers=1)`.

4. **NAS-Safe Database State (No Phantom States):**
   - `aiosqlite` is used in WAL mode. NEVER map this DB to an NFS/SMB network share.
   - Deduplication uses a composite key of `inode` + `size`. NEVER use `device_id` (volatile on network mounts).
   - IMMEDIATELY update DB state to `EXTRACTING` or `INFERENCING` the moment a file is popped off a queue. Do not wait for the pipeline to finish to update state.

5. **POSIX Atomic Subtitle Writing:**
   - NEVER write directly to `.srt`. Always write to `.movie.srt.tmp` first.
   - ALWAYS inherit `st_uid`, `st_gid`, and `st_mode` from the source MKV using `os.chown` and `os.chmod` on the temp file.
   - ALWAYS finalize with a true POSIX atomic swap: `os.replace(temp_srt_path, final_srt_path)`.

## 📐 CODE QUALITY & TESTING STANDARDS

- **Type Checking:** All code MUST pass `mypy src/aisrt tests --strict`. You cannot use `Any` unless absolutely necessary, and you must add comprehensive type hints.
- **Linting & Formatting:** All code MUST be formatted and linted via `ruff format .` and `ruff check .`.
- **Testing:** All new logic MUST be covered by `pytest` (using `pytest-asyncio` for async tests). Mock all I/O or network calls.
- **Style:** Use Google-style docstrings for all classes and methods. Follow existing idioms strictly.

## 📁 PROJECT STRUCTURE
- `src/aisrt/cli.py`: Typer CLI entrypoint.
- `src/aisrt/pipeline.py`: The `asyncio.TaskGroup` producer-consumer orchestrator.
- `src/aisrt/discovery.py`: NAS-safe file crawler (checks age, sibling SRTs, hardlinks, DB state, embedded streams).
- `src/aisrt/extractor.py`: Zero-disk FFmpeg stdout stream processor.
- `src/aisrt/hardware.py`: Profiler for CUDA, Apple Silicon, and CPU. Routes to the optimal Whisper model.
- `src/aisrt/stt.py`: Singleton worker encapsulating the Whisper model and `BatchedInferencePipeline`.
- `src/aisrt/assembly.py`: Broadcast-quality SRT chunker (by word timestamps and punctuation) and Atomic POSIX writer.
- `src/aisrt/state.py`: SQLite async state tracker.

## 🔧 AGENT BEHAVIOR
- Think silently. Think deeply. Evaluate edge cases for race conditions, memory leaks, and POSIX permission errors before writing any code.
- Provide surgical, minimal diffs. Do not rewrite unrelated code.
- If you change an architectural boundary, you MUST explicitly document the justification.