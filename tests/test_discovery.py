"""Tests for the DiscoveryEngine."""

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from srtgen.config import FilterConfig
from srtgen.discovery import DiscoveryEngine
from srtgen.state import FileState, StateTracker


@pytest.fixture
def filter_config() -> FilterConfig:
    """Provide a default FilterConfig."""
    return FilterConfig(
        min_age_mins=15,
        extensions=[".mkv", ".mp4"],
        exclude_patterns=["*sample*"],
        target_languages=["eng", "en"],
    )


@pytest.fixture
def temp_media_dir(tmp_path: Path) -> Path:
    """Create a temporary media directory with some files."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    # Create a valid file (older than 15 mins)
    valid_file = media_dir / "valid_movie.mkv"
    valid_file.write_text("dummy")
    old_time = time.time() - (30 * 60)
    os.utime(valid_file, (old_time, old_time))

    # Create a valid file with sibling srt
    sibling_file = media_dir / "has_sibling.mkv"
    sibling_file.write_text("dummy")
    os.utime(sibling_file, (old_time, old_time))
    (media_dir / "has_sibling.eng.srt").write_text("subtitle")

    # Create a recent file
    recent_file = media_dir / "recent_movie.mkv"
    recent_file.write_text("dummy")
    recent_time = time.time() - 60  # 1 min old
    os.utime(recent_file, (recent_time, recent_time))

    # Create a sample file (should be excluded)
    sample_file = media_dir / "movie_sample.mkv"
    sample_file.write_text("dummy")

    return media_dir


@pytest.mark.asyncio
async def test_scan_directory_basic(temp_media_dir: Path, filter_config: FilterConfig) -> None:
    """Test basic scanning, recency filtering, sibling checking, and exclusions."""
    mock_tracker = AsyncMock(spec=StateTracker)
    mock_tracker.get_state.return_value = None
    mock_tracker.check_hardlink_processed.return_value = False

    engine = DiscoveryEngine(temp_media_dir, filter_config, mock_tracker)

    with patch.object(engine, "_check_embedded_subtitles", new_callable=AsyncMock) as mock_embedded:
        mock_embedded.return_value = False

        results = []
        async for media_file, action in engine.scan():
            results.append((media_file.path.name, action))

        names_actions = dict(results)

        assert "valid_movie.mkv" in names_actions
        assert names_actions["valid_movie.mkv"] == "PROCESS"

        assert "has_sibling.mkv" in names_actions
        assert "SKIP: External sibling subtitle exists" in names_actions["has_sibling.mkv"]

        assert "recent_movie.mkv" in names_actions
        assert "SKIP: Modified recently" in names_actions["recent_movie.mkv"]

        assert "movie_sample.mkv" not in names_actions


@pytest.mark.asyncio
async def test_scan_database_state(temp_media_dir: Path, filter_config: FilterConfig) -> None:
    """Test that SQLite states correctly skip files."""
    mock_tracker = AsyncMock(spec=StateTracker)
    mock_tracker.check_hardlink_processed.return_value = False

    # Mock get_state to return COMPLETED for the valid movie
    completed_state = FileState(
        file_path=str(temp_media_dir / "valid_movie.mkv"),
        inode=1,
        device_id=1,
        mtime=1.0,
        size=5,  # size of "dummy"
        status="COMPLETED",
        model_used="model",
    )
    mock_tracker.get_state.return_value = completed_state

    engine = DiscoveryEngine(temp_media_dir, filter_config, mock_tracker)

    with patch.object(engine, "_check_embedded_subtitles", new_callable=AsyncMock) as mock_embedded:
        mock_embedded.return_value = False

        results = []
        async for media_file, action in engine.scan():
            if media_file.path.name == "valid_movie.mkv":
                results.append(action)

        assert len(results) == 1
        assert "SKIP: Already processed (Database)" in results[0]
