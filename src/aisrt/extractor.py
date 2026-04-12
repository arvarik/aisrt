"""Zero-Disk Audio Extraction Engine."""

import asyncio
import json
from pathlib import Path

import numpy as np
from loguru import logger


class AudioExtractor:
    """Extracts audio directly to memory (Zero-Disk) using FFmpeg."""

    @staticmethod
    async def get_audio_track_index(video_path: Path, target_languages: list[str]) -> int:
        """Use ffprobe to find the best audio track index.

        Prefers target languages (e.g., 'eng'). Falls back to the first audio track (0).

        Args:
            video_path: Path to the media file.
            target_languages: List of preferred ISO-639 language codes.

        Returns:
            The relative audio track index (e.g., 0 for the first audio track).
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index:stream_tags=language",
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
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                logger.warning(f"ffprobe failed for {video_path}, defaulting to track 0.")
                return 0

            data = json.loads(stdout.decode("utf-8"))
            streams = data.get("streams", [])

            # We iterate to find the relative index of the target language track.
            # ffmpeg's `-map 0:a:X` refers to the X-th stream, matching the list index here.
            for i, stream in enumerate(streams):
                tags = stream.get("tags", {})
                lang = tags.get("language", "").lower()
                if lang in target_languages:
                    logger.debug(f"Found preferred language '{lang}' at audio track {i}")
                    return i

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to probe audio tracks for {video_path}: {e}")

        # Default to the first audio stream
        return 0

    @staticmethod
    async def extract_audio_to_memory(
        video_path: Path, track_index: int, timeout: int = 1800
    ) -> np.ndarray:
        """Extract audio to a NumPy array via stdout.

        Forces 16kHz mono audio, required by Whisper, entirely in RAM.

        Args:
            video_path: Path to the media file.
            track_index: The relative audio track index to extract (e.g. 0).
            timeout: Maximum allowed time in seconds for FFmpeg execution.

        Returns:
            A normalized 32-bit float NumPy array.

        Raises:
            RuntimeError: If the FFmpeg process fails or times out.
        """
        logger.debug(f"Extracting audio track {track_index} from {video_path} into memory...")

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            f"0:a:{track_index}",
            "-vn",
            "-sn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ]

        # 500MB limit for the stdout buffer.
        # 16kHz mono 16-bit PCM = 32,000 bytes/sec.
        # 500MB = ~4.3 hours of uncompressed audio.
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # No limit imposed; we chunk the stream
        )

        audio_bytes = bytearray()
        stderr_bytes = bytearray()

        async def _read_stdout() -> None:
            if not process.stdout:
                return
            while chunk := await process.stdout.read(65536):
                audio_bytes.extend(chunk)

        async def _read_stderr() -> None:
            if not process.stderr:
                return
            while chunk := await process.stderr.read(65536):
                stderr_bytes.extend(chunk)

        try:
            # Concurrently read both streams to prevent pipeline blocking
            await asyncio.wait_for(
                asyncio.gather(_read_stdout(), _read_stderr(), process.wait()), timeout=timeout
            )
        except TimeoutError as err:
            process.kill()
            await process.communicate()
            raise RuntimeError(
                f"FFmpeg extraction timed out after {timeout}s for {video_path}"
            ) from err

        if process.returncode != 0:
            error_msg = stderr_bytes.decode().strip()
            raise RuntimeError(f"FFmpeg extraction failed for {video_path}: {error_msg}")

        if not audio_bytes:
            raise RuntimeError(f"FFmpeg extraction resulted in empty output for {video_path}")

        # Convert raw 16-bit PCM bytes to 32-bit float normalized between -1.0 and 1.0
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        return audio_np
