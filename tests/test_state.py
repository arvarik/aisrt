"""Tests for the StateTracker sqlite manager."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aisrt.state import StateTracker


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fixture providing a temporary database path."""
    return tmp_path / "test_state.db"


@pytest.mark.asyncio
async def test_state_tracker_close(db_path: Path) -> None:
    """Test that close() cleans up the connection."""
    tracker = StateTracker(db_path)

    # Use a mock connection to verify close() is called
    mock_conn = AsyncMock()
    tracker._conn = mock_conn

    await tracker.close()

    mock_conn.close.assert_awaited_once()
    assert tracker._conn is None

    # Test idempotency - closing again should not raise error
    await tracker.close()
    assert tracker._conn is None


@pytest.mark.asyncio
async def test_state_tracker_init(db_path: Path) -> None:
    """Test that the tracker initializes and creates tables correctly."""
    async with StateTracker(db_path) as tracker:
        assert db_path.exists()

        # Verify tables created
        assert tracker._conn is not None
        async with tracker._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='file_state';"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "file_state"


@pytest.mark.asyncio
async def test_update_and_get_state(db_path: Path) -> None:
    """Test inserting and retrieving a state."""
    async with StateTracker(db_path) as tracker:
        await tracker.update_state(
            file_path="/movies/movie.mkv",
            inode=12345,
            mtime=1000.0,
            size=5000,
            status="COMPLETED",
            model_used="large-v3-turbo",
        )

        state = await tracker.get_state("/movies/movie.mkv")
        assert state is not None
        assert state.file_path == "/movies/movie.mkv"
        assert state.inode == 12345
        assert state.size == 5000
        assert state.status == "COMPLETED"
        assert state.model_used == "large-v3-turbo"


@pytest.mark.asyncio
async def test_check_hardlink_processed(db_path: Path) -> None:
    """Test the inode/size deduplication logic."""
    async with StateTracker(db_path) as tracker:
        await tracker.update_state(
            file_path="/movies/movie_link1.mkv",
            inode=999,
            mtime=1000.0,
            size=5000,
            status="COMPLETED",
        )

        # Should be true because we inserted the exact inode/size pair
        assert await tracker.check_hardlink_processed(999, 5000) is True

        # Should be false for different inode or size
        assert await tracker.check_hardlink_processed(888, 5000) is False
        assert await tracker.check_hardlink_processed(999, 4000) is False

        # If it's merely PENDING, we haven't successfully processed it yet
        await tracker.update_state(
            file_path="/movies/movie_link2.mkv",
            inode=777,
            mtime=1000.0,
            size=5000,
            status="PENDING",
        )
        assert await tracker.check_hardlink_processed(777, 5000) is False


@pytest.mark.asyncio
async def test_reset_stale_states(db_path: Path) -> None:
    """Test that stuck states are reverted to PENDING on startup."""
    # First, insert stuck states
    async with StateTracker(db_path) as tracker:
        await tracker.update_state(
            file_path="/movies/1.mkv", inode=1, mtime=0, size=0, status="EXTRACTING"
        )
        await tracker.update_state(
            file_path="/movies/2.mkv", inode=2, mtime=0, size=0, status="INFERENCING"
        )
        await tracker.update_state(
            file_path="/movies/3.mkv", inode=3, mtime=0, size=0, status="COMPLETED"
        )

    # Re-connect to trigger startup logic (reset_stale_states is called in connect)
    async with StateTracker(db_path) as tracker:
        s1 = await tracker.get_state("/movies/1.mkv")
        s2 = await tracker.get_state("/movies/2.mkv")
        s3 = await tracker.get_state("/movies/3.mkv")

        assert s1 is not None and s1.status == "PENDING"
        assert s2 is not None and s2.status == "PENDING"
        assert s3 is not None and s3.status == "COMPLETED"  # Should remain unchanged
