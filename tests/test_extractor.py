"""Tests for the zero-disk AudioExtractor."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from srtgen.extractor import AudioExtractor


@pytest.mark.asyncio
async def test_get_audio_track_index_found() -> None:
    """Test ffprobe correctly parsing the JSON and finding 'eng' track."""
    mock_ffprobe_output = json.dumps(
        {
            "streams": [
                {"index": 0, "tags": {"language": "jpn"}},
                {"index": 1, "tags": {"language": "eng"}},
                {"index": 2, "tags": {"language": "fre"}},
            ]
        }
    ).encode("utf-8")

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        process_mock = AsyncMock()
        process_mock.communicate.return_value = (mock_ffprobe_output, b"")
        process_mock.returncode = 0
        mock_exec.return_value = process_mock

        track_idx = await AudioExtractor.get_audio_track_index(Path("dummy.mkv"), ["eng", "en"])
        assert track_idx == 1


@pytest.mark.asyncio
async def test_get_audio_track_index_fallback() -> None:
    """Test fallback to track 0 if preferred language is not found."""
    mock_ffprobe_output = json.dumps(
        {
            "streams": [
                {"index": 0, "tags": {"language": "jpn"}},
                {"index": 1, "tags": {"language": "fre"}},
            ]
        }
    ).encode("utf-8")

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        process_mock = AsyncMock()
        process_mock.communicate.return_value = (mock_ffprobe_output, b"")
        process_mock.returncode = 0
        mock_exec.return_value = process_mock

        track_idx = await AudioExtractor.get_audio_track_index(Path("dummy.mkv"), ["eng", "en"])
        assert track_idx == 0


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
