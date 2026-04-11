# Ultimate SRT Generator: Architecture & Implementation Strategy v2.0
**Status:** Approved for Production Implementation (Sr. Principal Engineer Edition)

## 1. Overview & Objectives
A highly resilient, hardware-aware, concurrent pipeline designed to crawl massive network-attached media libraries, perform zero-disk audio extraction, and generate broadcast-standard `.srt` subtitles using SOTA AI models.

**Design Philosophy:** Safe failure, zero network degradation, strict bounded memory usage, and precise POSIX permission inheritance.

---

## 2. Tech Stack & Core Libraries
*   **Language:** Python 3.11+ (Strict requirement for `asyncio.TaskGroup` for safe concurrent task cancellation).
*   **CLI & UX Framework:** `Typer`, `Rich`, `Loguru` (Rotated, bounded JSON logging).
*   **Configuration:** `Pydantic v2` (Strict bounds validation, frozen settings models).
*   **STT Engine:** `faster-whisper` (CTranslate2 backend).
*   **Media Processing:** Native `asyncio.create_subprocess_exec` executing static `ffmpeg`/`ffprobe` binaries.
*   **State Tracking:** `aiosqlite` (Async SQLite3 wrapper).
*   **Hardware Profiling:** `pynvml` (NVIDIA VRAM precision without importing PyTorch) and `psutil`.

---

## 3. Core Architecture & Resiliency Design

### 3.1. Bounded Producer-Consumer Pipeline (Memory Safe)
We utilize an `asyncio.TaskGroup` with strictly bounded queues to prevent OOM errors:
1.  **Scanner Task (I/O Bound):** Crawls paths, queries DB, runs `ffprobe` for embedded English subtitles. Pushes to `ExtractionQueue` (maxsize = CPU_CORES).
2.  **Extractor Workers (CPU Bound):** Pops path, streams FFmpeg to memory, yields NumPy array to `InferenceQueue` (maxsize = 2). The queue limit forces extraction to pause until the GPU is ready.
3.  **Inference Worker (GPU/CPU Singleton):** Pops NumPy array, runs AI within a `ThreadPoolExecutor(max_workers=1)`, writes atomic SRT.

### 3.2. Advanced Hardware Auto-Detection Matrix
The `ProfileEngine` initializes safely on startup to map the compute fabric:
*   **Thread Safety:** Explicitly inject `os.environ["OMP_NUM_THREADS"] = "1"` before importing libraries to prevent C-level thread thrashing.
*   **VRAM > 4GB (CUDA):** Load `large-v3-turbo` (`compute_type="float16"`).
*   **VRAM 2GB - 4GB (CUDA):** Load `large-v3-turbo` (`compute_type="int8_float16"`).
*   **Apple Silicon / CPU-only (> 16GB RAM):** Load `large-v3-turbo` (`compute_type="int8"`). Set `cpu_threads` to `psutil.cpu_count(logical=False)`.
*   **CPU-only (< 16GB RAM):** Load `small.en` (`compute_type="int8"`).

### 3.3. NAS-Safe Discovery & Deduplication
*   **Active File Protection:** Skip if `st_mtime` is within the last 15 minutes (avoids parsing actively downloading/remuxing MKVs).
*   **Embedded Stream Detection:** Execute `ffprobe -v error -select_streams s -show_entries stream=index:stream_tags=language -of json`. If an English subtitle stream exists, mark DB `STATUS=EMBEDDED_EXISTS` and skip.
*   **Inode Deduplication:** Track `st_ino` to avoid parsing hardlinks created by tools like Radarr/Sonarr multiple times.

### 3.4. Zero-Disk Audio Extraction Pipeline
Extracting audio into a memory pipe eliminates SSD wear and reduces I/O latency.
```python
# Execute async ffprobe first to find the primary English track (or fallback to track 0)
# Then extract directly to stdout:
cmd = [
    "ffmpeg", "-y", "-v", "error",
    "-i", source_path,
    "-map", f"0:a:{selected_track_index}", 
    "-vn", "-sn",              # Drop video/subs aggressively to save network bandwidth
    "-ac", "1",                # Downmix to Mono
    "-ar", "16000",            # Whisper sample rate requirement
    "-f", "s16le",             # Raw 16-bit PCM format
    "-"                        # Stream to stdout
]
# Wrap in asyncio.wait_for(timeout=1800) to kill hanging ffmpeg processes on corrupt MKVs
```
**Memory Conversion:** Capture stdout into `audio_bytes`, then convert:
`audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0`

### 3.5. STT Inference Guardrails & Broadcast Formatting
**Anti-Hallucination Constraints:**
```python
segments, info = model.transcribe(
    audio_np,
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500), # strict VAD gating
    condition_on_previous_text=False,                 # CRITICAL: Hard-stops looping text
    compression_ratio_threshold=2.4,                  # Drops zlib-compressible babble
    no_speech_threshold=0.6,
    word_timestamps=True,                             # Required for UX chunking
    initial_prompt="A well-punctuated English subtitle."
)
```
**Broadcast Subtitle Chunker:** Iterate through `segment.words`. Aggregate words until the string length exceeds ~42 characters or hits a terminal punctuation mark (`.`, `?`, `!`). Yield as a SubRip timecode block (`HH:MM:SS,mmm --> HH:MM:SS,mmm`).

### 3.6. True POSIX Atomic Assembly & Inheritance
1.  **Hidden Write:** Write chunked segments to `/mnt/nas/movies/.video_name.srt.tmp`.
2.  **Metadata Inheritance:** Read the `st_uid`, `st_gid`, and `st_mode` of the source `video.mkv`.
3.  **Apply Permissions:** Execute `os.chown(uid, gid)` and `os.chmod(mode)` on the temp `.srt`.
4.  **Atomic Commit:** `os.replace("/mnt/nas/movies/.video_name.srt.tmp", "/mnt/nas/movies/video_name.en.srt")`. This guarantees a 100% atomic swap with zero risk of leaving corrupt files if power is lost.

---

## 4. Phased Implementation Plan & Strict Coding Standards
**Engineering Mandate:** All code MUST adhere to `Ruff` formatting, pass `mypy --strict` type checking, and feature comprehensive Google-style docstrings.

### Phase 1: Core Scaffolding, Models & Database
1.  Initialize with Poetry (`pyproject.toml`).
2.  Define Pydantic v2 Configuration schemas (`AppConfig`).
3.  Implement `StateTracker` using `aiosqlite`.
4.  Path strictly bound to `~/.config/aisrt/state.db`.
5.  Execute `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` upon connection.

### Phase 2: Hardware Profiler & AI Routing
1.  Build `HardwareProfiler` (`pynvml` and `psutil`).
2.  Implement `ModelRouter` to resolve the matrix logic. Initialize the `faster-whisper` model.
3.  Implement the globally accessible `ThreadPoolExecutor(max_workers=1)` for the STT Worker.

### Phase 3: NAS-Safe Discovery Engine
1.  Implement async filesystem crawler using `os.scandir`.
2.  Implement async `ffprobe` wrapper to detect track indices and embedded subtitles.
3.  Wire the `--dry-run` CLI flag to output a highly detailed Rich TUI table (File | Size | Action | Reason).

### Phase 4: Zero-Disk Extractor & TaskGroups
1.  Implement `AudioExtractor` with the optimized FFmpeg stdout pipeline and numpy conversion.
2.  Wire the `asyncio.Queue` mechanism inside an `asyncio.TaskGroup()`. Implement the strict `maxsize` backpressure limits.

### Phase 5: Thread-Safe STT & Atomic Assembly
1.  Implement `STTWorker`. Offload transcription via `asyncio.get_running_loop().run_in_executor()`.
2.  Implement `SRTFormatter` with the 42-character word-timestamp chunking logic.
3.  Implement the Cross-Device safe `AtomicWriter` (`os.stat` inheritance -> write `.tmp` -> `os.chown`/`os.chmod` -> `os.replace`).

### Phase 6: Graceful Shutdown & Containerization
1.  Implement a `GracefulShutdown` context manager. Trap `SIGINT`/`SIGTERM`. Cancel the Scanner, wait for the Extractor to finish its current RAM buffer, flush the final SRT, and cleanly close the database connection.
2.  Create a multi-stage `Dockerfile` based on `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` (CTranslate2 requires cuDNN for maximum performance). Include `tini` as the entrypoint for proper process reaping. Document the `--shm-size=4gb` requirement for Docker runs to prevent CTranslate2 shared-memory core dumps.

---

👑 **Final Engineering Sign-off:**
By moving away from blocking synchronicity, discarding disk-based audio extraction for zero-latency memory buffers, and strictly enforcing POSIX permissions, this architecture transitions from a fragile script into a hardened enterprise utility capable of processing petabyte-scale libraries autonomously.