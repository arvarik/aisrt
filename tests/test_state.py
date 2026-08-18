"""Tests for the SQLite state tracker."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path

import pytest

from aisrt.state import (
    SCHEMA_VERSION,
    STATUS_COMPLETED,
    STATUS_EMBEDDED_EXISTS,
    STATUS_EXTRACTING,
    STATUS_FAILED,
    STATUS_INFERENCING,
    STATUS_PENDING,
    StateTracker,
    build_row,
)


@pytest.fixture
async def tracker(tmp_path: Path) -> AsyncIterator[StateTracker]:
    """Provide an open tracker backed by a temporary database."""
    async with StateTracker(tmp_path / "state.db") as store:
        yield store


class TestLifecycle:
    """Opening and closing the store must be safe and idempotent."""

    @pytest.mark.asyncio
    async def test_creates_the_database_and_schema(self, tmp_path: Path) -> None:
        """The parent directory, the table, and the version are all created."""
        db_path = tmp_path / "nested" / "dir" / "state.db"
        async with StateTracker(db_path):
            pass

        assert db_path.exists()
        with closing(sqlite3.connect(db_path)) as raw:
            assert raw.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            names = {row[0] for row in raw.execute("SELECT name FROM sqlite_master")}
        assert "file_state" in names
        assert "idx_file_state_dedup" in names
        assert "idx_file_state_status" in names

    @pytest.mark.asyncio
    async def test_using_a_closed_tracker_raises(self, tmp_path: Path) -> None:
        """Every query guards against a closed connection."""
        store = StateTracker(tmp_path / "state.db")
        with pytest.raises(RuntimeError, match="Database connection not established"):
            await store.get_state("anything")

    @pytest.mark.asyncio
    async def test_closing_twice_is_safe(self, tmp_path: Path) -> None:
        """A second close is a no-op, which matters during shutdown."""
        store = StateTracker(tmp_path / "state.db")
        await store.connect()
        await store.close()
        await store.close()


class TestReadWrite:
    """Rows round-trip through the store unchanged."""

    @pytest.mark.asyncio
    async def test_insert_then_read(self, tracker: StateTracker) -> None:
        """A written row comes back with every field intact."""
        await tracker.update_state(
            "/media/movie.mkv", 42, 1.5, 1024, STATUS_COMPLETED, "large-v3", device=7
        )
        state = await tracker.get_state("/media/movie.mkv")

        assert state is not None
        assert state.status == STATUS_COMPLETED
        assert state.model_used == "large-v3"
        assert state.inode == 42
        assert state.device == 7
        assert state.size == 1024

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, tracker: StateTracker) -> None:
        """A path that was never written is not an error."""
        assert await tracker.get_state("/media/absent.mkv") is None

    @pytest.mark.asyncio
    async def test_an_intermediate_status_keeps_the_model(self, tracker: StateTracker) -> None:
        """Recording progress must not erase which model produced a subtitle."""
        await tracker.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_COMPLETED, "large-v3")
        await tracker.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_EXTRACTING)

        state = await tracker.get_state("/m/a.mkv")
        assert state is not None
        assert state.model_used == "large-v3"

    @pytest.mark.asyncio
    async def test_batch_write(self, tracker: StateTracker) -> None:
        """Many rows commit in one transaction."""
        rows = [
            build_row(f"/m/{index}.mkv", index, 0, 1.0, 100, STATUS_EMBEDDED_EXISTS)
            for index in range(50)
        ]
        await tracker.update_states(rows)

        records = await tracker.get_skip_records()
        assert len(records) == 50

    @pytest.mark.asyncio
    async def test_an_empty_batch_is_a_no_op(self, tracker: StateTracker) -> None:
        """Writing nothing must not open a transaction."""
        await tracker.update_states([])
        assert await tracker.get_skip_records() == {}


class TestDeduplication:
    """Hardlinks must be transcribed once, and only once."""

    @pytest.mark.asyncio
    async def test_finished_content_is_recognised(self, tracker: StateTracker) -> None:
        """The same device, inode, and size counts as already done."""
        await tracker.update_state("/m/a.mkv", 99, 1.0, 500, STATUS_COMPLETED, device=3)
        assert await tracker.check_hardlink_processed(99, 500, device=3) is True

    @pytest.mark.asyncio
    async def test_a_different_device_is_a_different_file(self, tracker: StateTracker) -> None:
        """Two volumes can reuse an inode number without sharing content."""
        await tracker.update_state("/m/a.mkv", 99, 1.0, 500, STATUS_COMPLETED, device=3)
        assert await tracker.check_hardlink_processed(99, 500, device=4) is False

    @pytest.mark.asyncio
    async def test_unfinished_work_is_not_deduplicated(self, tracker: StateTracker) -> None:
        """A file still being extracted is not a reason to skip its hardlink."""
        await tracker.update_state("/m/a.mkv", 99, 1.0, 500, STATUS_EXTRACTING, device=3)
        assert await tracker.check_hardlink_processed(99, 500, device=3) is False

    @pytest.mark.asyncio
    async def test_bulk_identities(self, tracker: StateTracker) -> None:
        """The scan preloads every finished identity in one query."""
        await tracker.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_COMPLETED, device=1)
        await tracker.update_state("/m/b.mkv", 2, 1.0, 20, STATUS_EMBEDDED_EXISTS, device=1)
        await tracker.update_state("/m/c.mkv", 3, 1.0, 30, STATUS_FAILED, device=1)

        assert await tracker.get_all_processed_hardlinks() == {(1, 1, 10), (1, 2, 20)}


class TestSkipRecords:
    """The scan loads only the columns it reads."""

    @pytest.mark.asyncio
    async def test_prefix_filter(self, tracker: StateTracker) -> None:
        """A shared database does not load rows for another library."""
        await tracker.update_state("/media/movies/a.mkv", 1, 1.0, 10, STATUS_COMPLETED)
        await tracker.update_state("/media/shows/b.mkv", 2, 1.0, 20, STATUS_COMPLETED)

        records = await tracker.get_skip_records("/media/movies")
        assert set(records) == {"/media/movies/a.mkv"}

    @pytest.mark.asyncio
    async def test_wildcards_in_a_path_are_escaped(self, tracker: StateTracker) -> None:
        """A directory named with a percent sign must not act as a wildcard."""
        await tracker.update_state("/media/100%/a.mkv", 1, 1.0, 10, STATUS_COMPLETED)
        await tracker.update_state("/media/other/b.mkv", 2, 1.0, 20, STATUS_COMPLETED)

        records = await tracker.get_skip_records("/media/100%")
        assert set(records) == {"/media/100%/a.mkv"}

    @pytest.mark.asyncio
    async def test_attempts_are_tracked(self, tracker: StateTracker) -> None:
        """A repeatedly failing file records how often it was tried."""
        await tracker.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_FAILED, attempts=3)
        records = await tracker.get_skip_records()
        assert records["/m/a.mkv"].attempts == 3


class TestStaleRecovery:
    """A crash must not leave a file stuck forever."""

    @pytest.mark.asyncio
    async def test_transient_statuses_reset(self, tmp_path: Path) -> None:
        """Reopening the store returns in-flight files to pending."""
        db_path = tmp_path / "state.db"
        async with StateTracker(db_path) as store:
            await store.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_EXTRACTING)
            await store.update_state("/m/b.mkv", 2, 1.0, 20, STATUS_INFERENCING)
            await store.update_state("/m/c.mkv", 3, 1.0, 30, STATUS_COMPLETED)

        async with StateTracker(db_path) as store:
            states = await store.get_all_states()

        assert states["/m/a.mkv"].status == STATUS_PENDING
        assert states["/m/b.mkv"].status == STATUS_PENDING
        assert states["/m/c.mkv"].status == STATUS_COMPLETED


def _write_legacy_database(db_path: Path, *, not_null: bool) -> None:
    """Create a database in the schema an earlier build shipped."""
    constraint = "NOT NULL" if not_null else ""
    with closing(sqlite3.connect(db_path)) as raw, raw:
        raw.execute(
            f"""
            CREATE TABLE file_state (
                file_path TEXT PRIMARY KEY,
                inode INTEGER NOT NULL,
                device_id INTEGER {constraint},
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                status TEXT NOT NULL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        raw.execute(
            "INSERT INTO file_state (file_path, inode, device_id, mtime, size, status, "
            "model_used) VALUES ('/m/old.mkv', 7, 42, 1.0, 500, 'COMPLETED', 'large-v3')"
        )


class TestMigration:
    """Upgrading must never throw a user's history away, or block later writes."""

    @pytest.mark.parametrize("not_null", [True, False])
    @pytest.mark.asyncio
    async def test_an_old_schema_keeps_its_rows(self, tmp_path: Path, not_null: bool) -> None:
        """A database written by an earlier version is upgraded in place."""
        db_path = tmp_path / "state.db"
        _write_legacy_database(db_path, not_null=not_null)

        async with StateTracker(db_path) as store:
            state = await store.get_state("/m/old.mkv")

        assert state is not None, "the upgrade destroyed the user's history"
        assert state.status == STATUS_COMPLETED
        assert state.model_used == "large-v3"
        assert state.attempts == 0
        assert state.device == 42, "the device number was not carried across"

    @pytest.mark.parametrize("not_null", [True, False])
    @pytest.mark.asyncio
    async def test_writes_work_after_the_upgrade(self, tmp_path: Path, not_null: bool) -> None:
        """The obsolete column must be gone, not merely ignored.

        SQLite cannot drop a NOT NULL column, so an upgrade that only adds the
        new columns leaves a constraint that rejects every later insert.
        """
        db_path = tmp_path / "state.db"
        _write_legacy_database(db_path, not_null=not_null)

        async with StateTracker(db_path) as store:
            await store.update_state(
                "/m/new.mkv", 1, 1.0, 10, STATUS_COMPLETED, "tiny.en", device=3
            )
            await store.update_state("/m/old.mkv", 7, 2.0, 500, STATUS_COMPLETED, None, device=42)

            assert len(await store.get_all_states()) == 2
            refreshed = await store.get_state("/m/old.mkv")
            assert refreshed is not None
            assert refreshed.model_used == "large-v3"

    @pytest.mark.asyncio
    async def test_the_obsolete_column_is_removed(self, tmp_path: Path) -> None:
        """A rebuild retires the column rather than leaving it in place."""
        db_path = tmp_path / "state.db"
        _write_legacy_database(db_path, not_null=True)

        async with StateTracker(db_path):
            pass

        with closing(sqlite3.connect(db_path)) as raw:
            columns = {row[1] for row in raw.execute("PRAGMA table_info(file_state)")}
        assert "device_id" not in columns
        assert {"device", "attempts"} <= columns

    @pytest.mark.asyncio
    async def test_upgrading_twice_changes_nothing(self, tmp_path: Path) -> None:
        """Reopening an already-upgraded database is a no-op."""
        db_path = tmp_path / "state.db"
        _write_legacy_database(db_path, not_null=True)

        async with StateTracker(db_path) as store:
            await store.update_state("/m/new.mkv", 1, 1.0, 10, STATUS_COMPLETED, "tiny.en")
        async with StateTracker(db_path) as store:
            assert len(await store.get_all_states()) == 2

    @pytest.mark.asyncio
    async def test_reopening_is_idempotent(self, tmp_path: Path) -> None:
        """Running the migration twice changes nothing."""
        db_path = tmp_path / "state.db"
        async with StateTracker(db_path) as store:
            await store.update_state("/m/a.mkv", 1, 1.0, 10, STATUS_COMPLETED, "large-v3")
        async with StateTracker(db_path) as store:
            state = await store.get_state("/m/a.mkv")
        assert state is not None
        assert state.model_used == "large-v3"


class TestPurge:
    """Rows for deleted media are cleaned up on request."""

    @pytest.mark.asyncio
    async def test_removes_only_missing_paths(self, tracker: StateTracker) -> None:
        """A path still on disk is kept, a path that is gone is deleted."""
        await tracker.update_state("/m/keep.mkv", 1, 1.0, 10, STATUS_COMPLETED)
        await tracker.update_state("/m/gone.mkv", 2, 1.0, 20, STATUS_COMPLETED)

        removed = await tracker.purge_missing(["/m/keep.mkv"])

        assert removed == 1
        assert await tracker.get_state("/m/gone.mkv") is None
        assert await tracker.get_state("/m/keep.mkv") is not None
