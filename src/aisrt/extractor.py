"""Zero-disk audio extraction straight into a NumPy buffer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger

from aisrt.probing import FFmpegNotFoundError, reap_process

SAMPLE_RATE: Final = 16_000
"""Sample rate Whisper requires."""

_READ_CHUNK: Final = 1 << 20
"""Bytes read from the ffmpeg pipe at a time. Larger than the 64 KiB default
so that a two-hour film costs a few hundred transport resumes, not thousands."""

_STREAM_LIMIT: Final = 1 << 20
_STDERR_LIMIT: Final = 64 * 1024
_INV_INT16: Final = np.float32(1.0 / 32768.0)
_SLACK_SECONDS: Final = 2.0

MAX_DURATION_SECONDS: Final = 24 * 3600.0
"""Refuse to preallocate for a file whose header claims more than one day."""


class AudioExtractor:
    """Extracts audio from a container directly into memory."""

    @staticmethod
    async def extract_audio_to_memory(
        video_path: Path,
        track_index: int,
        timeout: float = 1800.0,
        duration: float | None = None,
    ) -> np.ndarray:
        """Decode one audio track to a 16 kHz mono float32 array.

        The samples never touch the disk. FFmpeg writes raw 16-bit PCM to a pipe
        and this method scales it into a buffer that was sized from the probed
        duration, so peak memory is one float32 array rather than a byte buffer
        plus a converted copy.

        Args:
            video_path: The media file to read.
            track_index: The relative audio track index, as used by ``-map 0:a:N``.
            timeout: Seconds to wait for FFmpeg before the file is abandoned.
            duration: The probed duration in seconds. It only sizes the initial
                buffer, so an approximate value is fine and None is allowed.

        Returns:
            A float32 array of samples in the range -1.0 to 1.0.

        Raises:
            RuntimeError: If FFmpeg fails, times out, or produces too few samples.
            FFmpegNotFoundError: If FFmpeg is not installed.
        """
        logger.debug(f"Extracting audio track {track_index} from {video_path.name}")

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            # Skip corrupt packets rather than emitting garbage samples.
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "crccheck",
            # Audio decoding is single threaded. Extra threads only fight the
            # inference worker for CPU.
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-map",
            f"0:a:{track_index}",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-max_error_rate",
            "0.5",
            "-",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            raise FFmpegNotFoundError(
                "ffmpeg not found on PATH. Install FFmpeg and try again."
            ) from error

        if process.stdout is None or process.stderr is None:  # pragma: no cover
            await reap_process(process)
            raise RuntimeError(f"FFmpeg pipes were not created for {video_path}")

        stdout, stderr = process.stdout, process.stderr
        stderr_bytes = bytearray()

        async def _read_stderr() -> None:
            while chunk := await stderr.read(65536):
                if len(stderr_bytes) < _STDERR_LIMIT:
                    stderr_bytes.extend(chunk[: _STDERR_LIMIT - len(stderr_bytes)])

        audio: np.ndarray = np.empty(0, dtype=np.float32)

        async def _read_stdout() -> None:
            nonlocal audio
            audio = await _drain_pcm(stdout, duration)

        try:
            async with asyncio.timeout(timeout):
                await asyncio.gather(_read_stdout(), _read_stderr())
                returncode = await process.wait()
        except TimeoutError:
            await reap_process(process)
            raise RuntimeError(
                f"FFmpeg extraction timed out after {timeout:.0f}s for {video_path}"
            ) from None
        except asyncio.CancelledError:
            await asyncio.shield(reap_process(process))
            raise
        finally:
            await reap_process(process)

        if returncode != 0:
            detail = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg extraction failed for {video_path}: {detail}")
        if audio.size < SAMPLE_RATE:
            raise RuntimeError(
                f"FFmpeg produced only {audio.size} samples for {video_path}. "
                "The audio track is missing or unreadable."
            )
        return audio


async def _drain_pcm(stdout: asyncio.StreamReader, duration: float | None) -> np.ndarray:
    """Read signed 16-bit PCM from a pipe into a preallocated float32 buffer.

    Args:
        stdout: The pipe carrying raw ``s16le`` samples.
        duration: The probed duration in seconds, used to size the buffer.

    Returns:
        A float32 view holding exactly the samples that were read.
    """
    buffer = np.empty(_initial_capacity(duration), dtype=np.float32)
    written = 0
    carry = b""

    while True:
        chunk = await stdout.read(_READ_CHUNK)
        if not chunk:
            break
        if carry:
            chunk = carry + chunk
            carry = b""
        if len(chunk) % 2:
            # A sample can straddle a pipe boundary. Keep the odd byte back.
            carry, chunk = chunk[-1:], chunk[:-1]
            if not chunk:
                continue

        samples = np.frombuffer(chunk, dtype=np.int16)
        if written + samples.size > buffer.size:
            grown = np.empty(max(buffer.size * 2, written + samples.size), dtype=np.float32)
            grown[:written] = buffer[:written]
            buffer = grown
        np.multiply(
            samples,
            _INV_INT16,
            out=buffer[written : written + samples.size],
            dtype=np.float32,
            casting="unsafe",
        )
        written += samples.size

    if carry:
        logger.debug("Discarding a trailing partial PCM sample.")

    # Return a view, not a copy. Copying would briefly hold both arrays and
    # raise peak memory by half again. The caller uses resident_bytes() to
    # account for the tail that the view keeps alive.
    return buffer[:written]


def resident_bytes(audio: np.ndarray) -> int:
    """Report the memory an array actually holds.

    A slice of a larger buffer is a view. Its ``nbytes`` counts only the visible
    samples, while the whole allocation stays resident until the view is dropped,
    so the memory budget must account for the base.

    Args:
        audio: The array to measure.

    Returns:
        The size in bytes of the allocation that keeps this array alive.
    """
    base = audio.base
    if isinstance(base, np.ndarray):
        return int(base.nbytes)
    return int(audio.nbytes)


def _initial_capacity(duration: float | None) -> int:
    """Choose the starting size of the sample buffer.

    Args:
        duration: The probed duration in seconds, or None when it is unknown.

    Returns:
        A sample count. Unknown or implausible durations fall back to a small
        buffer that grows geometrically.
    """
    if duration is None or duration <= 0:
        return SAMPLE_RATE * 60
    capped = min(duration, MAX_DURATION_SECONDS)
    if capped < duration:
        logger.warning(
            f"Container reports a duration of {duration:.0f}s. "
            f"Sizing the audio buffer for {capped:.0f}s instead."
        )
    return int((capped + _SLACK_SECONDS) * SAMPLE_RATE)
