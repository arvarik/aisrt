"""Tests for the zero-disk AudioExtractor."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from aisrt.extractor import AudioExtractor


@pytest.mark.asyncio
async def test_extract_audio_to_memory() -> None:
    """Test FFmpeg stdout capture into numpy array."""
    # Generate some fake 16-bit PCM data
    # 2 samples of 16-bit audio = 4 bytes
    fake_pcm_data = b"\x00\x40\x00\xc0"  # Values: 16384, -16384

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        process_mock = AsyncMock()

        # Mock stdout.read() to return data once, then empty bytes to signify EOF
        process_mock.stdout.read.side_effect = [fake_pcm_data, b""]
        process_mock.stderr.read.side_effect = [b""]

        process_mock.returncode = 0
        mock_exec.return_value = process_mock

        audio_np = await AudioExtractor.extract_audio_to_memory(Path("dummy.mkv"), 0)

        assert isinstance(audio_np, np.ndarray)
        assert audio_np.dtype == np.float32
        assert len(audio_np) == 2
        # 16384 / 32768.0 = 0.5
        # -16384 / 32768.0 = -0.5
        np.testing.assert_allclose(audio_np, np.array([0.5, -0.5], dtype=np.float32))
