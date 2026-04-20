"""SQLite state tracker for media files processing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger


@dataclass
class FileState:
    """Represents the tracking state of a specific media file."""

    file_path: str
    inode: int
    mtime: float
    size: int
    status: str
    model_used: str | None = None
    timestamp: str | None = None


class StateTracker:
    """Asynchronous SQLite manager for tracking generation progress.

    Ensures safe operation across multiple runs, tracks inode deduplication,
    and avoids parsing files already marked COMPLETED or EMBEDDED_EXISTS.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the state tracker.

        Args:
            db_path: Path to the SQLite database.
        """
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "StateTracker":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Establish the database connection and apply PRAGMAs."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)

        # Enable Write-Ahead Logging for better concurrency and safety over local mounts
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")

        await self.setup()
        await self.reset_stale_states()

    async def setup(self) -> None:
        """Create the necessary tables if they do not exist."""
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        # Dropping table if it exists with the old schema (device_id)
        # to ensure clean upgrade for new users.
        try:
            async with self._conn.execute("PRAGMA table_info(file_state);") as cursor:
                columns = await cursor.fetchall()
                if columns and any(col[1] == "device_id" for col in columns):
                    logger.info("Upgrading database schema: removing volatile device_id.")
                    await self._conn.execute("DROP TABLE file_state")
        except Exception as e:
            logger.debug(f"Schema upgrade check skipped: {e}")

        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_state (
                file_path TEXT PRIMARY KEY,
                inode INTEGER NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                status TEXT NOT NULL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Index on inode/size for quick hardlink deduplication checks
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_state_inode_size ON file_state (inode, size)"
        )
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get_state(self, file_path: str) -> FileState | None:
        """Retrieve the state of a given file by path.

        Args:
            file_path: The absolute path of the media file.

        Returns:
            The FileState object if it exists, otherwise None.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = (
            "SELECT file_path, inode, mtime, size, status, model_used, timestamp "
            "FROM file_state WHERE file_path = ?"
        )
        async with self._conn.execute(query, (file_path,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return FileState(*row)
        return None

    async def check_hardlink_processed(self, inode: int, size: int) -> bool:
        """Check if an identical inode/size pair has already been completed.

        Args:
            inode: The file's inode.
            size: File size in bytes.

        Returns:
            True if this exact file data has been successfully processed under any path.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = (
            "SELECT 1 FROM file_state "
            "WHERE inode = ? AND size = ? "
            "AND status IN ('COMPLETED', 'EMBEDDED_EXISTS')"
        )
        async with self._conn.execute(query, (inode, size)) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def get_all_processed_hardlinks(self) -> set[tuple[int, int]]:
        """Fetch all inode/size pairs that have already been processed.

        Returns:
            A set of (inode, size) tuples.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = (
            "SELECT inode, size FROM file_state WHERE status IN ('COMPLETED', 'EMBEDDED_EXISTS')"
        )
        async with self._conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            return {(row[0], row[1]) for row in rows}

    async def get_all_states(self) -> dict[str, FileState]:
        """Fetch all tracked file states.

        Returns:
            A dictionary mapping file paths to FileState objects.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = (
            "SELECT file_path, inode, mtime, size, status, model_used, timestamp FROM file_state"
        )
        async with self._conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            return {row[0]: FileState(*row) for row in rows}

    async def update_state(
        self,
        file_path: str,
        inode: int,
        mtime: float,
        size: int,
        status: str,
        model_used: str | None = None,
    ) -> None:
        """Update or insert the state of a media file.

        Args:
            file_path: The absolute path of the media file.
            inode: The file's inode.
            mtime: Last modified time.
            size: File size in bytes.
            status: Processing status (PENDING, EXTRACTING, COMPLETED, etc.).
            model_used: The STT model string used if completed.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = """
            INSERT INTO file_state (
                file_path, inode, mtime, size, status, model_used, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_path) DO UPDATE SET
                inode=excluded.inode,
                mtime=excluded.mtime,
                size=excluded.size,
                status=excluded.status,
                model_used=excluded.model_used,
                timestamp=CURRENT_TIMESTAMP
        """
        await self._conn.execute(
            query,
            (file_path, inode, mtime, size, status, model_used),
        )
        await self._conn.commit()

    async def reset_stale_states(self) -> None:
        """Reset any EXTRACTING or INFERENCING status back to PENDING.

        This prevents files from being permanently stuck if the daemon crashed.
        """
        if not self._conn:
            raise RuntimeError("Database connection not established.")

        query = (
            "UPDATE file_state SET status = 'PENDING' WHERE status IN ('EXTRACTING', 'INFERENCING')"
        )
        async with self._conn.execute(query) as cursor:
            if cursor.rowcount > 0:
                logger.info(
                    f"Reset {cursor.rowcount} stale file states to PENDING "
                    "after unexpected shutdown."
                )
        await self._conn.commit()
