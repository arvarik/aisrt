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

*   **Zero-Disk Audio Extraction:** Stops SSD wear-and-tear by bypassing `/tmp/` files completely. Audio streams are asynchronously ripped via FFmpeg and piped directly into RAM (NumPy arrays) for AI ingestion.
*   **Bounded Asynchronous Concurrency:** Eliminates Python GIL starvation via an `asyncio.TaskGroup` producer-consumer pipeline. It extracts audio streams exactly as fast as the GPU can transcribe them, strictly capping memory usage.
*   **Intelligent Hardware Routing Matrix:** Auto-detects NVIDIA GPUs (VRAM), Apple Silicon, or pure CPU environments to intelligently route to the most optimal `large-v3-turbo`, `small`, or quantized `int8` model.
*   **NAS-Safe & Deduplicating:** Backed by an asynchronous local SQLite database (WAL mode) tracking `inode` and `size`. It avoids parsing active downloads, seamlessly skips duplicate hardlinks, and performs strict POSIX Atomic `os.replace` operations with original MKV metadata inheritance.
*   **Broadcast Formatting:** Implements a strict chunking algorithm on top of Whisper's word-level timestamps. No more "walls of text"—subtitles are limited to ~42 chars and 2 lines, naturally breaking on terminal punctuation.

---

## 🚀 Installation & Deployment

### 🐳 Docker (Recommended for TrueNAS / Unraid)

For maximum stability and ease-of-use with NVIDIA hardware, use the provided Docker stack.

1.  Clone the repository:
    ```bash
    git clone https://github.com/arvarik/srt-generator.git
    cd srt-generator
    ```
2.  Review and modify the `docker-compose.yml` to point to your media directory:
    ```yaml
    volumes:
      - /mnt/user/media/movies:/media:rw
      - ./srtgen_data:/root/.config/srtgen:rw
    ```
3.  Deploy:
    ```bash
    docker compose up --build -d
    ```

### 💻 Native Python (Ubuntu Desktop / Server / macOS)

**Prerequisites:** Python 3.11+ and `ffmpeg` must be installed on your system.

```bash
# 1. Clone repository
git clone https://github.com/arvarik/srt-generator.git
cd srt-generator

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

### Dry-Run (Scan)
Safely scan a directory to see exactly what hardware will be loaded and what files will be processed, without actually running the AI model.

```bash
srtgen scan /path/to/movies --min-age-mins 60 --verbose
```

### Live Run
Execute the extraction and inference pipeline.

```bash
srtgen run /path/to/movies
```

### CLI Overrides & Environment Variables
You can manually override the hardware auto-detector and execution options.

**Via CLI:**
```bash
srtgen run /path/to/movies --force-device cuda --force-model large-v3-turbo --translate --watch --watch-interval 60
```

**Via Docker Environment Variables:**
Since `AppConfig` utilizes `pydantic-settings`, you can configure the daemon entirely through your `docker-compose.yml`:
*   `SRTGEN_TRANSLATE=True` (Auto-dub foreign audio into English)
*   `SRTGEN_WATCH=True` (Run 24/7 as a daemon)
*   `SRTGEN_WATCH_INTERVAL_MINS=60` (Time between library scans)
*   `SRTGEN_FILTERS__MIN_AGE_MINS=30` (Skip active torrent/usenet downloads)
*   `SRTGEN_HARDWARE__FORCE_MODEL=large-v3-turbo`

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
mypy src/srtgen tests  # Strict Type Checking
pytest tests           # Asynchronous Unit Tests
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
