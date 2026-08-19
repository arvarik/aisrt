<div align="center">
  <h1>🎬 aisrt</h1>
  <p><strong>Hardware-aware, zero-disk, concurrent AI pipeline for mass-generating broadcast-quality subtitles.</strong></p>

  [![CI](https://github.com/arvarik/aisrt/actions/workflows/ci.yml/badge.svg)](https://github.com/arvarik/aisrt/actions/workflows/ci.yml)
  [![PyPI](https://img.shields.io/pypi/v/aisrt)](https://pypi.org/project/aisrt/)
  [![Python](https://img.shields.io/pypi/pyversions/aisrt)](https://pypi.org/project/aisrt/)
  [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

---

## Overview

`aisrt` crawls a media library, finds the videos that have no subtitle, and writes one
for each of them. It extracts audio straight into memory, transcribes it with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and commits a `.srt` file
next to the video.

It is built for a NAS: it never writes temporary audio to disk, it caps how much memory
it holds, it remembers what it has already done, and it survives a restart in the middle
of a run.

### What it does

- **Zero-disk audio extraction.** FFmpeg streams raw PCM into a preallocated NumPy
  buffer. Nothing lands on the SSD. Supports `.mkv`, `.mp4`, `.avi`, `.webm`, `.ts`,
  `.m2ts`, and `.vob`.
- **Broadcast-standard subtitles.** Cues follow the Netflix Timed Text Style Guide:
  42 characters per line, at most 2 lines, at most 20 characters per second, a minimum
  duration of 5/6 second, a maximum of 7 seconds, and a two-frame gap between cues. Lines
  break at natural points and never separate an article from its noun.
- **Hardware-aware routing.** The tool reads the GPU, its VRAM, and the system memory,
  then picks a model, a device, and a precision that fit. It validates the precision
  against CTranslate2 before it loads anything.
- **Bounded memory.** The pipeline caps decoded audio in bytes, not in file count, so a
  three-hour film cannot push the process into the out-of-memory killer.
- **Resumable state.** An asynchronous SQLite database records every file. It
  deduplicates hardlinks by device and inode, skips files that already have a subtitle,
  and returns interrupted files to the queue after a crash.
- **Graceful shutdown.** `SIGINT` and `SIGTERM` finish the current file, flush the
  database, and exit with a clear status. A second signal stops immediately.

---

## Installation

FFmpeg must be installed and on `PATH`. `aisrt` checks for it at startup and reports a
clear error if it is missing.

### Python

```bash
uv tool install aisrt   # Recommended
pipx install aisrt      # Isolated environment
pip install aisrt       # Standard
```

### Docker

For an NVIDIA GPU, the provided image carries CUDA 12.9 with cuDNN 9, which CTranslate2
requires.

1. Clone the repository:
   ```bash
   git clone https://github.com/arvarik/aisrt.git
   cd aisrt
   ```
2. Point `docker-compose.yml` at your media directory:
   ```yaml
   volumes:
     - /mnt/user/media/movies:/media:rw
     - ./aisrt_data:/config
   ```
3. Start it:
   ```bash
   docker compose up -d
   ```

The container runs as UID 1000, so the subtitles it writes belong to your user rather
than to root.

---

## Usage

### Check what a run would do

```bash
aisrt scan /mnt/movies --min-age-mins 60
```

Crawls the library, skips files modified in the last 60 minutes, ignores videos that
already carry a subtitle, and prints a table of what it would process.

### Transcribe a library

```bash
aisrt run /mnt/movies
```

Extracts audio to memory, profiles the hardware, loads the best model that fits, writes
a `.srt` next to each video, and records progress in the state database.

### Translate foreign audio into English

```bash
aisrt run /mnt/anime --translate
```

Whisper reads the original audio and writes English subtitles. The router picks a model
that can translate. The turbo and `*.en` checkpoints cannot translate, so `--translate`
routes to `large-v3` or `medium` instead.

### Run as a daemon

```bash
aisrt run /mnt/media --watch --watch-interval 30
```

Scans, processes, sleeps for 30 minutes, and scans again. `SIGTERM` wakes it immediately
and shuts it down cleanly, so `docker stop` and `systemctl stop` both behave.

### Force a small model

```bash
aisrt run /mnt/media --force-device cpu --force-model small.en
```

Bypasses the router. Useful on a weak machine, or to reproduce a result exactly.

### Trade accuracy for throughput

```bash
aisrt run /mnt/media --batch-size 16
```

Batched inference is roughly three times faster. It also discards the temperature
fallback and the hallucination guard that catch repetition loops on long films, so
`aisrt` decodes sequentially by default.

---

## Command reference

Both commands take the media directory as their only argument.

| Option | `scan` | `run` | Meaning |
| --- | :---: | :---: | --- |
| `--min-age-mins` | yes | yes | Skip files modified within this many minutes. Default 15. |
| `--ext` | yes | yes | Media extension to accept. Repeatable. |
| `--exclude` | yes | yes | Glob pattern to skip, matched against each path component, ignoring case. Repeatable. |
| `--lang` | yes | yes | Subtitle language to generate and look for. Repeatable. The first names the file. |
| `--db-path` | yes | yes | Path of the state database. |
| `--limit` | yes | no | Rows to show in the scan report. Default 200. |
| `--translate` | no | yes | Translate speech into English. |
| `--language` | no | yes | Force the spoken language, for example `ja`. Detected when absent. |
| `--watch` | no | yes | Keep running and rescan on an interval. |
| `--watch-interval` | no | yes | Minutes between scans. Default 60. |
| `--max-memory-mb` | no | yes | Cap on decoded audio held in memory. Default 2048. |
| `--force-device` | no | yes | `cuda`, `cpu`, or `auto`. |
| `--force-model` | no | yes | Model alias or a local model directory. |
| `--force-compute-type` | no | yes | `float16`, `int8_float16`, or `int8`. |
| `--batch-size` | no | yes | Decode this many chunks together. |
| `--max-chars-per-line` | no | yes | Characters per subtitle line. Default 42. |
| `--max-cps` | no | yes | Reading speed ceiling. Default 20. |
| `--verbose`, `-v` | yes | yes | Debug logging. |
| `--version` | — | — | Print the version and exit. |

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every file was processed or deliberately skipped. |
| `1` | At least one file failed, or the pipeline itself failed. |
| `2` | The configuration is wrong, for example a missing directory or no FFmpeg. |
| `130` | A signal stopped the run. |

A cron job can therefore tell a clean run from a run that lost files.

### Environment variables

Every setting has an `AISRT_` variable. A command line option overrides it; when the
option is absent the variable takes effect.

| Variable | Meaning |
| --- | --- |
| `AISRT_TRANSLATE` | Translate speech into English. |
| `AISRT_WATCH` | Run as a daemon. |
| `AISRT_WATCH_INTERVAL_MINS` | Minutes between scans. |
| `AISRT_LANGUAGE` | Force the spoken language. |
| `AISRT_DB_PATH` | Path of the state database. |
| `AISRT_MAX_MEMORY_MB` | Cap on decoded audio held in memory. |
| `AISRT_FILTERS__MIN_AGE_MINS` | Skip files modified within this many minutes. |
| `AISRT_FILTERS__TARGET_LANGUAGES` | Subtitle languages, as a JSON list. |
| `AISRT_HARDWARE__FORCE_MODEL` | Model alias or local directory. |
| `AISRT_HARDWARE__FORCE_DEVICE` | `cuda`, `cpu`, or `auto`. |
| `AISRT_HARDWARE__FORCE_COMPUTE_TYPE` | Compute precision. |
| `AISRT_SUBTITLES__MAX_CPS` | Reading speed ceiling. |
| `HF_TOKEN` | Hugging Face token, to bypass the anonymous download rate limit. |

---

## Model routing

The router reads the hardware and the task, then picks the largest model that fits.

| Condition | Model | Device | Precision |
| --- | --- | --- | --- |
| CUDA, VRAM ≥ 10 GB | `large-v3` | cuda | `float16` |
| CUDA, VRAM ≥ 8 GB | `large-v3` | cuda | `int8_float16` |
| CUDA, VRAM ≥ 6 GB | `large-v3-turbo` | cuda | `float16` |
| CUDA, VRAM ≥ 4 GB | `large-v3-turbo` | cuda | `int8_float16` |
| CUDA, VRAM < 4 GB | `small.en` | cuda | `int8_float16` |
| No CUDA, RAM ≥ 16 GB | `large-v3-turbo` | cpu | `int8` |
| No CUDA, RAM < 16 GB | `small.en` | cpu | `int8` |

With `--translate`, every row that would pick `large-v3-turbo` picks `medium` instead,
and every row that would pick `small.en` picks `small`. The turbo checkpoint and the
English-only checkpoints return the original language when asked to translate.

**Apple Silicon runs on the CPU.** CTranslate2 has no Metal backend, so there is no GPU
path on a Mac. The `int8` precision is the fastest option there, because it is the only
one that uses more than one core.

---

## State database

The database lives at `${XDG_CONFIG_HOME:-~/.config}/aisrt/state.db`. Keep it on local
storage: SQLite write-ahead logging does not work over NFS or SMB, and `aisrt` warns when
it detects that.

One table, `file_state`, records the path, the device and inode, the size, the modified
time, the status, the model that produced the subtitle, and how many attempts a file has
had.

| Status | Meaning |
| --- | --- |
| `PENDING` | Queued, or returned to the queue after an interrupted run. |
| `EXTRACTING` | FFmpeg is decoding the audio. |
| `INFERENCING` | The model is transcribing. |
| `COMPLETED` | A subtitle was written. |
| `NO_SPEECH` | The model found no speech. The file is not retried. |
| `EMBEDDED_EXISTS` | The container already carries a text subtitle. |
| `FAILED` | The file failed. It is retried up to three times across runs. |

To start over, delete the database:

```bash
rm ~/.config/aisrt/state.db*
```

To retry only the failures:

```bash
sqlite3 ~/.config/aisrt/state.db "DELETE FROM file_state WHERE status = 'FAILED';"
```

---

## Troubleshooting

**`ffmpeg not found on PATH`.** Install FFmpeg. On Debian or Ubuntu,
`sudo apt install ffmpeg`. On macOS, `brew install ffmpeg`.

**`Could not load library libcudnn_ops_infer.so.8`.** CTranslate2 4.5 and later need
cuDNN 9. Upgrade the CUDA image, or run with `--force-device cpu`.

**The subtitles are in the wrong language after `--translate`.** The model was not
trained for translation. Add `--force-model large-v3` or `--force-model medium`. The tool
warns about this at startup.

**Hugging Face rate limits during the first run.** Set `HF_TOKEN` to a read token.

**`SQLite refused write-ahead logging`.** The database is on a network share. Move it
with `--db-path /local/path/state.db`.

**A repeated caption loop on one film.** Batched inference discards the guards that catch
this. Drop `--batch-size` so the run decodes sequentially.

---

## Development

```bash
uv sync              # Install the project and its development group
uv run pytest        # Tests, with coverage
uv run ruff check .  # Lint
uv run ruff format . # Format
uv run mypy          # Strict type check
```

Integration tests run the real FFmpeg binary. They skip themselves when it is not
installed. Run only those with `uv run pytest -m integration`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
