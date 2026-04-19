"""Tests for the probing utilities."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aisrt.probing import get_audio_track_index, has_embedded_subtitles, has_external_subtitle


@pytest.mark.asyncio
async def test_has_embedded_subtitles_positive() -> None:
    """Test detecting target-language text subtitles."""
    mock_data = {
        "streams": [
            {"codec_name": "subrip", "tags": {"language": "eng"}},
            {"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
        ]
    }

    with patch("aisrt.probing._run_ffprobe", new_callable=AsyncMock) as mock_probe:
        mock_probe.return_value = mock_data
        result = await has_embedded_subtitles(Path("dummy.mkv"), ["eng"])
        assert result is True


@pytest.mark.asyncio
async def test_has_embedded_subtitles_negative() -> None:
    """Test ignoring image-based or wrong-language subtitles."""
    mock_data = {
        "streams": [
            {"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
            {"codec_name": "subrip", "tags": {"language": "fra"}},
        ]
    }

    with patch("aisrt.probing._run_ffprobe", new_callable=AsyncMock) as mock_probe:
        mock_probe.return_value = mock_data
        result = await has_embedded_subtitles(Path("dummy.mkv"), ["eng"])
        assert result is False


@pytest.mark.asyncio
async def test_get_audio_track_index_preferred() -> None:
    """Test finding the correct language track."""
    mock_data = {
        "streams": [
            {"index": 0, "tags": {"language": "fra"}},
            {"index": 1, "tags": {"language": "eng"}},
        ]
    }

    with patch("aisrt.probing._run_ffprobe", new_callable=AsyncMock) as mock_probe:
        mock_probe.return_value = mock_data
        index = await get_audio_track_index(Path("dummy.mkv"), ["eng"])
        assert index == 1


@pytest.mark.asyncio
async def test_get_audio_track_index_default() -> None:
    """Test defaulting to track 0 if no match."""
    mock_data = {
        "streams": [
            {"index": 0, "tags": {"language": "jpn"}},
        ]
    }

    with patch("aisrt.probing._run_ffprobe", new_callable=AsyncMock) as mock_probe:
        mock_probe.return_value = mock_data
        index = await get_audio_track_index(Path("dummy.mkv"), ["eng"])
        assert index == 0


def test_has_external_subtitle_variants(tmp_path: Path) -> None:
    """Test different SRT naming patterns."""
    video = tmp_path / "movie.mkv"
    video.write_text("dummy")

    # 1. No subtitle
    assert has_external_subtitle(video, ["eng"]) is False

    # 2. Generic .srt
    srt = tmp_path / "movie.srt"
    srt.write_text("sub")
    assert has_external_subtitle(video, ["eng"]) is True
    srt.unlink()

    # 3. Language-specific .eng.srt
    eng_srt = tmp_path / "movie.eng.srt"
    eng_srt.write_text("sub")
    assert has_external_subtitle(video, ["eng"]) is True
