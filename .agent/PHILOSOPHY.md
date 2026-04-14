# Product Philosophy

_This is the soul of the product. It explains why the app exists and what its core beliefs are. Product Visionaries and UI/UX Designers use this to make feature and design decisions. Engineers use it to resolve ambiguity._

## 1. Why This Exists
I have a massive media library with missing subtitles. Manually generating SRTs is tedious and SSD-wear intensive. This tool autonomously processes entire libraries with zero intervention, zero disk wear, and highly optimized hardware orchestration.

## 2. Target User
This is for NAS hoarders and sysadmins who manage petabyte-scale media libraries. They are comfortable with Docker, CLI tools, and cron jobs. The application must be "set-and-forget" stable for 24/7 daemon modes.

## 3. Core Beliefs
- **Zero-Disk SSD Preservation**: Extracting terabytes of audio should not wear down SSD lifespans. We stream everything to memory.
- **Hardware Awareness**: Maximize hardware utilization natively without configuration tweaking. Route automatically to the best Whisper model that fits in VRAM/RAM.
- **Resilience**: The daemon must run indefinitely without GIL deadlocks, OOM errors, or database corruption from network dropouts.

## 4. Design & UX Principles
- **Beautiful Terminal Output**: Rich, informative progress bars and dashboard summaries.
- **Safe by Default**: Do not mutate or overwrite the source media files. Use atomic file swapping and retain MKV metadata attributes.

## 5. What This Is NOT
- Not a general-purpose text-to-speech UI app.
- Not a video transcoder/converter.
