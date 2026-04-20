"""Shared media probing and subtitle discovery utilities."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger


async def _run_ffprobe(video_path: Path, select_streams: str, show_entries: str) -> dict[str, Any]:
    """Internal helper to run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        select_streams,
        "-show_entries",
        show_entries,
        "-of",
        "json",
        str(video_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"ffprobe failed for {video_path}: {stderr.decode().strip()}")
            return {}

        result: dict[str, Any] = json.loads(stdout.decode("utf-8"))
        return result
    except FileNotFoundError:
        logger.error("ffprobe not found. Please ensure FFmpeg is installed and in PATH.")
        raise
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse ffprobe JSON for {video_path}")
        return {}
    except Exception as e:
        logger.warning(f"Error probing {video_path}: {e}")
        return {}


async def has_embedded_subtitles(video_path: Path, target_languages: list[str]) -> bool:
    """Check if the video contains text-based embedded subtitles in target languages."""
    data = await _run_ffprobe(video_path, "s", "stream=index,codec_name:stream_tags=language")
    streams = data.get("streams", [])

    for stream in streams:
        codec = stream.get("codec_name", "").lower()
        tags = stream.get("tags", {})
        lang = tags.get("language", "").lower()

        if lang in target_languages:
            # Only count text-based subtitles.
            # Image-based subs (hdmv_pgs_subtitle) force transcodes on many players.
            if codec in ["subrip", "ass", "mov_text", "webvtt"]:
                return True
            else:
                logger.debug(
                    f"Ignoring embedded {codec} subtitle in {video_path} (forces transcode)"
                )
    return False


async def get_audio_track_index(video_path: Path, target_languages: list[str]) -> int:
    """Find the best audio track index, preferring target languages."""
    data = await _run_ffprobe(video_path, "a", "stream=index:stream_tags=language")
    streams = data.get("streams", [])

    for i, stream in enumerate(streams):
        tags = stream.get("tags", {})
        lang = tags.get("language", "").lower()
        if lang in target_languages:
            logger.debug(f"Found preferred language '{lang}' at audio track {i}")
            return i

    return 0


def has_external_subtitle(video_path: Path, target_languages: list[str]) -> bool:
    """Check if an external SRT file exists next to the video."""
    base_name = video_path.stem
    dir_name = video_path.parent

    check_suffixes = [".srt"]
    for lang in target_languages:
        check_suffixes.append(f".{lang}.srt")

    return any((dir_name / f"{base_name}{suffix}").exists() for suffix in check_suffixes)
