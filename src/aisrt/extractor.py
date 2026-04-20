"""Zero-Disk Audio Extraction Engine."""

import asyncio
from pathlib import Path

import numpy as np
from loguru import logger


class AudioExtractor:
    """Extracts audio directly to memory (Zero-Disk) using FFmpeg."""

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
            error_msg = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg extraction failed for {video_path}: {error_msg}")

        if not audio_bytes:
            raise RuntimeError(f"FFmpeg extraction resulted in empty output for {video_path}")

        # Convert raw 16-bit PCM bytes to 32-bit float normalized between -1.0 and 1.0
        loop = asyncio.get_running_loop()

        def _convert_to_np() -> np.ndarray:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_np /= 32768.0
            return audio_np

        return await loop.run_in_executor(None, _convert_to_np)
