"""Media probing and subtitle discovery built on a single ffprobe call."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from loguru import logger

FFPROBE_TIMEOUT: Final = 60.0
"""Seconds to wait for ffprobe before the file is abandoned."""

PROBE_SIZE: Final = "5000000"
ANALYZE_DURATION: Final = "5000000"

TEXT_SUBTITLE_CODECS: Final = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608", "subviewer"}
)
"""Subtitle codecs a player can render without a video transcode."""

# ISO 639-2/B codes mapped to their ISO 639-1 form. Only the codes a media
# library realistically carries are listed.
_ISO_639_2_TO_1: Final = {
    "eng": "en",
    "fre": "fr",
    "fra": "fr",
    "ger": "de",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "jpn": "ja",
    "chi": "zh",
    "zho": "zh",
    "kor": "ko",
    "dut": "nl",
    "nld": "nl",
    "swe": "sv",
    "nor": "no",
    "dan": "da",
    "fin": "fi",
    "pol": "pl",
    "tur": "tr",
    "ara": "ar",
    "heb": "he",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "ces": "cs",
    "cze": "cs",
    "ell": "el",
    "gre": "el",
    "hun": "hu",
    "ukr": "uk",
    "ron": "ro",
    "rum": "ro",
    "bul": "bg",
    "cat": "ca",
    "ind": "id",
    "may": "ms",
    "msa": "ms",
}


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg toolchain is missing from PATH."""


def _as_float(value: object) -> float | None:
    """Convert a JSON scalar to a float, or return None when it is not numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalize_language(tag: str | None) -> str | None:
    """Reduce a language tag to a lowercase ISO 639-1 code.

    Args:
        tag: A raw tag from a media container, for example ``"eng"``,
            ``"en-US"``, or ``" EN "``.

    Returns:
        The two-letter code, or the cleaned tag when no mapping exists, or None
        when the tag carries no language.
    """
    if not tag:
        return None
    cleaned = tag.strip().lower().replace("_", "-").split("-")[0]
    if not cleaned or cleaned in {"und", "unknown", "none", "zxx", "mul"}:
        return None
    return _ISO_639_2_TO_1.get(cleaned, cleaned)


def normalize_languages(tags: list[str]) -> frozenset[str]:
    """Normalize a list of language tags and drop the ones that carry no code."""
    return frozenset(code for code in (normalize_language(tag) for tag in tags) if code)


@dataclass(frozen=True, slots=True)
class AudioTrack:
    """One audio stream, described by the fields that drive track selection."""

    index: int
    """Position among the audio streams. Use it with ``-map 0:a:{index}``."""

    language: str | None
    codec: str
    channels: int
    is_default: bool
    title: str | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Everything one ffprobe call tells us about a media file."""

    duration: float | None
    audio_tracks: tuple[AudioTrack, ...] = ()
    text_subtitle_languages: frozenset[str] = field(default_factory=frozenset)
    probe_failed: bool = False

    def has_text_subtitle(self, targets: frozenset[str]) -> bool:
        """Report whether a renderable subtitle exists in one of the targets."""
        return bool(self.text_subtitle_languages & targets)

    def select_audio_track(self, preferred: frozenset[str]) -> int:
        """Choose the audio track to transcribe.

        The order is: a track in a preferred language, then the container's
        default track, then the track with the most channels, then the first.

        Args:
            preferred: Normalized language codes to prefer. Pass an empty set to
                keep the container's own ordering, which is what translation
                needs because the original language track is the useful one.

        Returns:
            The relative audio track index for ``-map 0:a:{index}``.
        """
        if not self.audio_tracks:
            return 0
        if preferred:
            for track in self.audio_tracks:
                if track.language in preferred:
                    return track.index
        for track in self.audio_tracks:
            if track.is_default:
                return track.index
        return max(self.audio_tracks, key=lambda track: (track.channels, -track.index)).index


def require_ffmpeg() -> None:
    """Check that ffmpeg and ffprobe are on PATH.

    Raises:
        FFmpegNotFoundError: If either binary is missing.
    """
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FFmpegNotFoundError(
            f"{' and '.join(missing)} not found on PATH. "
            "Install FFmpeg and try again: https://ffmpeg.org/download.html"
        )


async def reap_process(process: asyncio.subprocess.Process, grace: float = 5.0) -> None:
    """Stop a child process and its group, then collect its exit status.

    The function never raises. A process wedged in uninterruptible I/O, which
    happens on a stalled NFS mount, is logged and left behind so that the
    pipeline keeps running.

    Args:
        process: The child process to stop.
        grace: Seconds to wait after each signal.
    """
    if process.returncode is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), sig)
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(grace):
                await process.wait()
                return
    logger.warning(f"Child process {process.pid} did not stop. Leaving it behind.")


async def _run_ffprobe(video_path: Path, timeout: float = FFPROBE_TIMEOUT) -> dict[str, Any]:
    """Run ffprobe once and return the parsed JSON document.

    Args:
        video_path: The media file to inspect.
        timeout: Seconds to wait before the probe is abandoned.

    Returns:
        The parsed document, or an empty dict when the probe failed.

    Raises:
        FFmpegNotFoundError: If ffprobe is not installed.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-hide_banner",
        "-analyzeduration",
        ANALYZE_DURATION,
        "-probesize",
        PROBE_SIZE,
        "-show_entries",
        (
            "format=duration:"
            "stream=index,codec_type,codec_name,channels:"
            "stream_tags=language,title:"
            "stream_disposition=default"
        ),
        "-of",
        "json",
        str(video_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        raise FFmpegNotFoundError(
            "ffprobe not found on PATH. Install FFmpeg and try again."
        ) from error

    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await process.communicate()
    except TimeoutError:
        await reap_process(process)
        logger.warning(f"ffprobe timed out after {timeout:.0f}s for {video_path}")
        return {}
    except asyncio.CancelledError:
        await asyncio.shield(reap_process(process))
        raise

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        logger.warning(f"ffprobe failed for {video_path}: {detail}")
        return {}

    try:
        parsed: dict[str, Any] = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"ffprobe returned invalid JSON for {video_path}")
        return {}
    return parsed


async def probe_media(video_path: Path, timeout: float = FFPROBE_TIMEOUT) -> MediaInfo:
    """Inspect a media file with a single ffprobe call.

    Args:
        video_path: The media file to inspect.
        timeout: Seconds to wait before the probe is abandoned.

    Returns:
        The duration, the audio tracks, and the languages of every renderable
        embedded subtitle. ``probe_failed`` is True when ffprobe produced
        nothing, which the caller must not read as "this file has no subtitles".

    Raises:
        FFmpegNotFoundError: If ffprobe is not installed.
    """
    document = await _run_ffprobe(video_path, timeout)
    if not document:
        return MediaInfo(duration=None, probe_failed=True)

    duration = _as_float(document.get("format", {}).get("duration"))
    if duration is not None and not (math.isfinite(duration) and duration > 0):
        duration = None

    audio_tracks: list[AudioTrack] = []
    subtitle_languages: set[str] = set()

    for stream in document.get("streams", []):
        codec_type = str(stream.get("codec_type", "")).lower()
        tags = stream.get("tags") or {}
        language = normalize_language(tags.get("language"))

        if codec_type == "audio":
            channels = int(_as_float(stream.get("channels")) or 0)
            audio_tracks.append(
                AudioTrack(
                    index=len(audio_tracks),
                    language=language,
                    codec=str(stream.get("codec_name", "")).lower(),
                    channels=channels,
                    is_default=bool((stream.get("disposition") or {}).get("default")),
                    title=tags.get("title"),
                )
            )
        elif codec_type == "subtitle":
            codec = str(stream.get("codec_name", "")).lower()
            if codec in TEXT_SUBTITLE_CODECS:
                if language:
                    subtitle_languages.add(language)
            else:
                # Image subtitles such as PGS force a transcode on many players,
                # so they do not count as an existing subtitle.
                logger.debug(f"Ignoring image subtitle ({codec}) in {video_path.name}")

    return MediaInfo(
        duration=duration,
        audio_tracks=tuple(audio_tracks),
        text_subtitle_languages=frozenset(subtitle_languages),
    )


def external_subtitle_path(video_path: Path, target_languages: frozenset[str]) -> Path | None:
    """Find a sidecar subtitle file that already sits next to the video.

    The check is synchronous. Call it from a worker thread, never from the event
    loop, because each probe is a network round trip on a NAS.

    Args:
        video_path: The media file.
        target_languages: Normalized language codes that count as a match.

    Returns:
        The path of the first matching subtitle, or None.
    """
    parent = video_path.parent
    stem = video_path.stem
    candidates = [f"{stem}.srt"]
    for language in sorted(target_languages):
        candidates.append(f"{stem}.{language}.srt")
        for alias, code in _ISO_639_2_TO_1.items():
            if code == language:
                candidates.append(f"{stem}.{alias}.srt")
    for candidate in candidates:
        path = parent / candidate
        if path.exists():
            return path
    return None


def has_external_subtitle(video_path: Path, target_languages: list[str]) -> bool:
    """Report whether a sidecar subtitle already exists next to the video."""
    return external_subtitle_path(video_path, normalize_languages(target_languages)) is not None
