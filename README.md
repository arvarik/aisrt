<div align="center">
  <h1>🎬 Ultimate SRT Generator</h1>
  <p><strong>Hardware-aware, zero-disk, highly concurrent AI pipeline for mass-generating broadcast-quality subtitles.</strong></p>

  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 🌟 Overview

The **Ultimate SRT Generator** is a production-grade daemon built for power users, NAS hoarders, and sysadmins. It autonomously crawls massive network-attached media libraries, detects videos missing English subtitles, performs **zero-disk** audio extraction directly into RAM, and infers broadcast-quality `.srt` files using state-of-the-art [faster-whisper](https://github.com/SYSTRAN/faster-whisper) AI models.

### 🛠️ Key Architectural Features

*   **Zero-Disk Audio Extraction:** Stops SSD wear-and-tear by bypassing `/tmp/` files completely. Audio streams are asynchronously ripped via FFmpeg and piped directly into RAM (NumPy arrays) for AI ingestion. Supports `.mkv`, `.mp4`, `.avi`, `.webm`, as well as broadcast and physical media formats like `.ts`, `.m2ts`, and `.vob`.
*   **Bounded Asynchronous Concurrency:** Eliminates Python GIL starvation via an `asyncio.TaskGroup` producer-consumer pipeline. It extracts audio streams exactly as fast as the GPU can transcribe them, strictly capping memory usage.
*   **Intelligent Hardware Routing Matrix:** Auto-detects NVIDIA GPUs (VRAM), Apple Silicon, or pure CPU environments to intelligently route to the most optimal `large-v3` (Ultra tier, >= 10GB VRAM), `large-v3-turbo`, `small`, or quantized `int8` model.
*   **NAS-Safe & Deduplicating:** Backed by an asynchronous local SQLite database (WAL mode) tracking `inode` and `size`. It avoids parsing active downloads, seamlessly skips duplicate hardlinks, and performs strict POSIX Atomic `os.replace` operations with original MKV metadata inheritance.
*   **Broadcast Formatting & Progress UI:** Implements a strict chunking algorithm on top of Whisper's word-level timestamps. Subtitles naturally break on terminal punctuation. The CLI features a beautiful Rich-based dynamic progress bar and a comprehensive end-of-run statistical dashboard.

---

## 🚀 Installation & Deployment

### 🐳 Docker (Recommended for TrueNAS / Unraid)

For maximum stability and ease-of-use with NVIDIA hardware, use the provided Docker stack.

1.  Clone the repository:
    ```bash
    git clone https://github.com/arvarik/aisrt.git
    cd aisrt
    ```
2.  Review and modify the `docker-compose.yml` to point to your media directory:
    ```yaml
    volumes:
      - /mnt/user/media/movies:/media:rw
      - ./aisrt_data:/root/.config/aisrt:rw
    ```
3.  Deploy:
    ```bash
    docker compose up -d
    ```

### 💻 Native Python (Ubuntu Desktop / Server / macOS)

**Prerequisites:** Python 3.11+ and `ffmpeg` must be installed on your system.

```bash
# 1. Clone repository
git clone https://github.com/arvarik/aisrt.git
cd aisrt

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the application
pip install -e .

# 4. (Optional) Install development dependencies
pip install -e ".[dev]"
```

---

## 🎮 Usage

The application features a beautifully formatted CLI built on `Typer` and `Rich`.

### 5 Robust Usage Examples

Here are five powerful ways to utilize `aisrt` in your homelab or production environment:

#### 1. The Safe Dry-Run (Check What Will Happen)
If you just pointed the tool at a massive 10TB media library and want to ensure it skips the right files without actually running the GPU inference:
```bash
aisrt scan /mnt/movies --min-age-mins 60
```
* **What it does:** Scans the library, skips files downloaded in the last 60 minutes, ignores videos that already have a `.srt` or an embedded English text track, and prints a beautiful table showing exactly which files will be processed.

#### 2. Standard Mass Generation
The standard run command. Perfect for a weekly cronjob.
```bash
aisrt run /mnt/movies
```
* **What it does:** Runs the full pipeline. Extracts audio to RAM, automatically profiles your hardware, boots the optimal Whisper model (e.g., `large-v3-turbo`), generates broadcast-standard atomic `.srt` files, and saves progress to the SQLite database.

#### 3. Foreign Cinema AI Translation (Dub to English)
You have a folder full of Anime or French cinema and want English subtitles generated.
```bash
aisrt run /mnt/anime --translate
```
* **What it does:** Same as the standard run, but it passes `task="translate"` into Whisper. The AI will ingest the foreign audio and natively dub it into perfect English `.srt` files!

#### 4. The 24/7 NAS Watchdog Daemon
You want `aisrt` to run in the background on your server and automatically process new movies as soon as they are added to your library.
```bash
aisrt run /mnt/media --watch --watch-interval 30
```
* **What it does:** Turns the CLI into a long-running daemon. It will scan the directory, process any missing subtitles, and then gracefully sleep for 30 minutes before waking up to check for new files again.

#### 5. Hardcoding Low-End Hardware
You are running this on a potato CPU or a server with a very weak GPU, and you want to ensure it doesn't try to load a heavy model, or you want to force CPU execution.
```bash
aisrt run /mnt/media --force-device cpu --force-model small.en
```
* **What it does:** Bypasses the auto-profiler and explicitly locks the execution to your CPU using the lightweight `small.en` quantized model.

### Docker Environment Variables
If you are deploying via Docker Compose, you can configure these exact same behaviors using `pydantic-settings` environment variables instead of CLI flags:
*   `HF_TOKEN=your_huggingface_token` (Bypass HuggingFace rate limits for model downloads)
*   `AISRT_TRANSLATE=True` (Auto-dub foreign audio into English)
*   `AISRT_WATCH=True` (Run 24/7 as a daemon)
*   `AISRT_WATCH_INTERVAL_MINS=60` (Time between library scans)
*   `AISRT_FILTERS__MIN_AGE_MINS=30` (Skip files currently being downloaded)
*   `AISRT_HARDWARE__FORCE_MODEL=large-v3-turbo`

---

## 🏗️ Open Source Development

We welcome contributions! The codebase strictly adheres to enterprise-level typing and styling.

**Development Setup:**
```bash
poetry install   # Or pip install -e ".[dev]"
```

**Running Tests & Linters:**
```bash
ruff check .           # Linter
ruff format .          # Formatter
mypy src/aisrt tests  # Strict Type Checking
pytest tests           # Asynchronous Unit Tests
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
