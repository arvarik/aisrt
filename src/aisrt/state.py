"""SQLite state tracker for media file processing."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import aiosqlite
from loguru import logger

T = TypeVar("T", bound=Callable[..., Any])

SCHEMA_VERSION: Final = 2
"""Current schema. ``PRAGMA user_version`` holds the version of an open file."""

STATUS_PENDING: Final = "PENDING"
STATUS_EXTRACTING: Final = "EXTRACTING"
STATUS_INFERENCING: Final = "INFERENCING"
STATUS_COMPLETED: Final = "COMPLETED"
STATUS_FAILED: Final = "FAILED"
STATUS_EMBEDDED_EXISTS: Final = "EMBEDDED_EXISTS"
STATUS_NO_SPEECH: Final = "NO_SPEECH"

TERMINAL_STATUSES: Final = (STATUS_COMPLETED, STATUS_EMBEDDED_EXISTS, STATUS_NO_SPEECH)
"""Statuses that mean the file needs no further work."""

TRANSIENT_STATUSES: Final = (STATUS_EXTRACTING, STATUS_INFERENCING)
"""Statuses that only a running pipeline should hold."""

# Fully literal SQL. Each IN list carries one placeholder per status constant
# above, so keep the counts in step if you add a status.
_HARDLINK_SQL: Final = (
    "SELECT 1 FROM file_state "
    "WHERE device = ? AND inode = ? AND size = ? AND status IN (?, ?, ?) LIMIT 1"
)
_FINISHED_IDENTITY_SQL: Final = (
    "SELECT device, inode, size FROM file_state WHERE status IN (?, ?, ?)"
)
_RESET_STALE_SQL: Final = "UPDATE file_state SET status = ? WHERE status IN (?, ?)"

_UPSERT_SQL: Final = """
    INSERT INTO file_state (
        file_path, inode, device, mtime, size, status, model_used, attempts, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(file_path) DO UPDATE SET
        inode=excluded.inode,
        device=excluded.device,
        mtime=excluded.mtime,
        size=excluded.size,
        status=excluded.status,
        model_used=COALESCE(excluded.model_used, file_state.model_used),
        attempts=excluded.attempts,
        timestamp=CURRENT_TIMESTAMP
"""


_CREATE_TABLE_SQL: Final = """
    CREATE TABLE IF NOT EXISTS file_state (
        file_path TEXT PRIMARY KEY,
        inode INTEGER NOT NULL,
        device INTEGER NOT NULL DEFAULT 0,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        status TEXT NOT NULL,
        model_used TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

_ADDED_COLUMNS: Final = {
    "device": "ALTER TABLE file_state ADD COLUMN device INTEGER NOT NULL DEFAULT 0",
    "attempts": "ALTER TABLE file_state ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
}

# Columns an earlier build created that the current schema does not use. Their
# presence forces a table rebuild, because SQLite cannot drop a NOT NULL column.
_OBSOLETE_COLUMNS: Final = frozenset({"device_id"})


def require_conn(method: T) -> T:
    """Ensure the database connection is open before the method runs."""

    @functools.wraps(method)
    async def wrapper(self: StateTracker, *args: Any, **kwargs: Any) -> Any:
        if self._conn is None:
            raise RuntimeError("Database connection not established.")
        return await method(self, *args, **kwargs)

    return cast(T, wrapper)


@dataclass(slots=True)
class FileState:
    """The tracked state of one media file."""

    file_path: str
    inode: int
    mtime: float
    size: int
    status: str
    model_used: str | None = None
    timestamp: str | None = None
    device: int = 0
    attempts: int = 0


@dataclass(frozen=True, slots=True)
class SkipRecord:
    """The subset of a row that the discovery filter reads."""

    status: str
    size: int
    mtime: float
    attempts: int


class StateTracker:
    """Asynchronous SQLite store for per-file processing state.

    The store survives restarts, deduplicates hardlinks by device and inode, and
    keeps files that already have subtitles out of the pipeline.
    """

    def __init__(self, db_path: Path, busy_timeout_ms: int = 10_000) -> None:
        """Initialize the tracker.

        Args:
            db_path: Path of the SQLite file. Its parent directory is created.
            busy_timeout_ms: Milliseconds SQLite waits for a lock before it
                reports that the database is busy.
        """
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> StateTracker:
        """Open the connection."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Close the connection."""
        await self.close()

    async def connect(self) -> None:
        """Open the database, apply the PRAGMAs, and run migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(
            self.db_path, timeout=self.busy_timeout_ms / 1000.0, isolation_level=None
        )

        async with self._conn.execute("PRAGMA journal_mode=WAL") as cursor:
            row = await cursor.fetchone()
        mode = str(row[0]).lower() if row else "unknown"
        if mode != "wal":
            # SQLite reports the mode it actually applied. A network mount has no
            # shared memory, so it silently stays in rollback mode.
            logger.warning(
                f"SQLite refused write-ahead logging at {self.db_path} (mode={mode}). "
                "Move the database off the network share. Writes are much slower now."
            )

        for pragma in (
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA busy_timeout={self.busy_timeout_ms}",
            "PRAGMA cache_size=-64000",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA wal_autocheckpoint=1000",
        ):
            await self._conn.execute(pragma)

        await self.setup()
        await self.reset_stale_states()

    @require_conn
    async def setup(self) -> None:
        """Create the table and bring an older file up to the current schema."""
        conn = self._connection()
        await conn.execute(_CREATE_TABLE_SQL)
        await self._migrate(conn)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_state_dedup ON file_state (device, inode, size)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_state_status ON file_state (status)"
        )
        await conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        await conn.commit()

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        """Bring an older database up to the current schema.

        The migration always preserves rows, so a user keeps the history of
        every file already transcribed. A column that no longer belongs is
        removed by rebuilding the table, because an obsolete ``NOT NULL`` column
        would reject every later insert.
        """
        columns = await self._column_names(conn)
        if not columns:
            return

        obsolete = columns & _OBSOLETE_COLUMNS
        if obsolete:
            logger.info(
                f"Upgrading the state database: rebuilding the table to drop "
                f"{', '.join(sorted(obsolete))}. Existing rows are kept."
            )
            await self._rebuild_table(conn, columns)
            return

        for column, statement in _ADDED_COLUMNS.items():
            if column not in columns:
                logger.info(f"Upgrading the state database: adding the '{column}' column.")
                await conn.execute(statement)

    @staticmethod
    async def _column_names(conn: aiosqlite.Connection) -> set[str]:
        """Return the column names of the state table, or an empty set."""
        async with conn.execute("PRAGMA table_info(file_state)") as cursor:
            rows = await cursor.fetchall()
        return {str(row[1]) for row in rows}

    @staticmethod
    async def _rebuild_table(conn: aiosqlite.Connection, columns: set[str]) -> None:
        """Copy every row into a table with the current schema.

        SQLite cannot drop a column that carries a ``NOT NULL`` constraint, so
        the only way to retire one is to build a new table and move the rows.
        """
        # An older build stored the device number under a different name. Carry
        # the value across when it is there, because it feeds hardlink
        # deduplication. A wrong value only costs a re-check, never a wrong skip
        # of a file that still needs a subtitle.
        if "device" in columns:
            device_expr = "COALESCE(device, 0)"
        elif "device_id" in columns:
            device_expr = "COALESCE(device_id, 0)"
        else:
            device_expr = "0"
        attempts_expr = "COALESCE(attempts, 0)" if "attempts" in columns else "0"

        await conn.execute("BEGIN IMMEDIATE")
        try:
            await conn.execute(_CREATE_TABLE_SQL.replace("file_state", "file_state_migrated", 1))
            # device_expr and attempts_expr are chosen above from a fixed set of
            # literals. No user value reaches this statement, and SQLite copies
            # the row values itself.
            await conn.execute(
                "INSERT INTO file_state_migrated "  # noqa: S608
                "(file_path, inode, device, mtime, size, status, model_used, attempts, timestamp) "
                f"SELECT file_path, inode, {device_expr}, mtime, size, status, model_used, "
                f"{attempts_expr}, timestamp FROM file_state"
            )
            await conn.execute("DROP TABLE file_state")
            await conn.execute("ALTER TABLE file_state_migrated RENAME TO file_state")
        except BaseException:
            await conn.rollback()
            raise
        await conn.commit()

    async def close(self) -> None:
        """Checkpoint the write-ahead log and close the connection."""
        if self._conn is None:
            return
        try:
            await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await self._conn.execute("PRAGMA optimize")
        except (aiosqlite.Error, ValueError) as error:
            logger.debug(f"Could not checkpoint the state database: {error}")
        await self._conn.close()
        self._conn = None

    def _connection(self) -> aiosqlite.Connection:
        """Return the open connection.

        Raises:
            RuntimeError: If the connection is closed.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not established.")
        return self._conn

    @require_conn
    async def get_state(self, file_path: str) -> FileState | None:
        """Read the state of one file.

        Args:
            file_path: The absolute path of the media file.

        Returns:
            The stored state, or None when the file is unknown.
        """
        query = (
            "SELECT file_path, inode, mtime, size, status, model_used, timestamp, "
            "device, attempts FROM file_state WHERE file_path = ?"
        )
        async with self._connection().execute(query, (file_path,)) as cursor:
            row = await cursor.fetchone()
        return FileState(*row) if row else None

    @require_conn
    async def check_hardlink_processed(self, inode: int, size: int, device: int = 0) -> bool:
        """Report whether the same file content was already finished.

        Args:
            inode: The inode of the file.
            size: The size in bytes.
            device: The device the file lives on. Two files on different devices
                may share an inode number without sharing content.

        Returns:
            True when a finished row exists for this content.
        """
        params = (device, inode, size, *TERMINAL_STATUSES)
        async with self._connection().execute(_HARDLINK_SQL, params) as cursor:
            return await cursor.fetchone() is not None

    @require_conn
    async def get_all_processed_hardlinks(self) -> set[tuple[int, int, int]]:
        """Fetch the content identity of every finished file.

        Returns:
            A set of ``(device, inode, size)`` tuples.
        """
        async with self._connection().execute(_FINISHED_IDENTITY_SQL, TERMINAL_STATUSES) as cursor:
            rows = await cursor.fetchall()
        return {(int(row[0]), int(row[1]), int(row[2])) for row in rows}

    @require_conn
    async def get_skip_records(self, path_prefix: str | None = None) -> dict[str, SkipRecord]:
        """Fetch the columns the discovery filter needs, and nothing else.

        Args:
            path_prefix: Restrict the result to paths under this prefix. Passing
                the media directory keeps a shared database from loading rows for
                libraries that are not being scanned.

        Returns:
            A mapping of file path to its skip record.
        """
        query = "SELECT file_path, status, size, mtime, attempts FROM file_state"
        params: tuple[Any, ...] = ()
        if path_prefix:
            query += " WHERE file_path LIKE ? ESCAPE '\\'"
            escaped = path_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params = (f"{escaped}%",)
        async with self._connection().execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return {
            str(row[0]): SkipRecord(
                status=str(row[1]), size=int(row[2]), mtime=float(row[3]), attempts=int(row[4])
            )
            for row in rows
        }

    @require_conn
    async def get_all_states(self) -> dict[str, FileState]:
        """Fetch every tracked state. Prefer ``get_skip_records`` for scanning."""
        query = (
            "SELECT file_path, inode, mtime, size, status, model_used, timestamp, "
            "device, attempts FROM file_state"
        )
        async with self._connection().execute(query) as cursor:
            rows = await cursor.fetchall()
        return {str(row[0]): FileState(*row) for row in rows}

    @require_conn
    async def update_state(
        self,
        file_path: str,
        inode: int,
        mtime: float,
        size: int,
        status: str,
        model_used: str | None = None,
        device: int = 0,
        attempts: int = 0,
    ) -> None:
        """Insert or update the state of one media file.

        Args:
            file_path: The absolute path of the media file.
            inode: The inode of the file.
            mtime: The last modification time.
            size: The size in bytes.
            status: One of the ``STATUS_*`` constants.
            model_used: The model that produced the subtitle. Passing None keeps
                the value already stored, so an intermediate transition does not
                erase it.
            device: The device the file lives on.
            attempts: How many times the pipeline has tried this file.
        """
        conn = self._connection()
        await conn.execute(
            _UPSERT_SQL,
            (file_path, inode, device, mtime, size, status, model_used, attempts),
        )
        await conn.commit()

    @require_conn
    async def update_states(self, rows: Sequence[tuple[Any, ...]]) -> None:
        """Insert or update many rows in one transaction.

        Args:
            rows: Tuples of ``(file_path, inode, device, mtime, size, status,
                model_used, attempts)``.
        """
        if not rows:
            return
        conn = self._connection()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            await conn.executemany(_UPSERT_SQL, rows)
        except BaseException:
            await conn.rollback()
            raise
        await conn.commit()

    @require_conn
    async def reset_stale_states(self) -> None:
        """Return files left mid-flight by a crash to the pending state."""
        conn = self._connection()
        params = (STATUS_PENDING, *TRANSIENT_STATUSES)
        async with conn.execute(_RESET_STALE_SQL, params) as cursor:
            reset_count = cursor.rowcount
        await conn.commit()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} file(s) left mid-run by an earlier shutdown.")

    @require_conn
    async def purge_missing(self, existing_paths: Iterable[str]) -> int:
        """Delete rows whose media file no longer exists.

        Args:
            existing_paths: Every path found by the current scan.

        Returns:
            The number of rows deleted.
        """
        conn = self._connection()
        keep = set(existing_paths)
        async with conn.execute("SELECT file_path FROM file_state") as cursor:
            rows = await cursor.fetchall()
        gone = [(str(row[0]),) for row in rows if str(row[0]) not in keep]
        if not gone:
            return 0
        await conn.execute("BEGIN IMMEDIATE")
        try:
            await conn.executemany("DELETE FROM file_state WHERE file_path = ?", gone)
        except BaseException:
            await conn.rollback()
            raise
        await conn.commit()
        logger.info(f"Removed {len(gone)} state row(s) for files that no longer exist.")
        return len(gone)


def build_row(
    file_path: str,
    inode: int,
    device: int,
    mtime: float,
    size: int,
    status: str,
    model_used: str | None = None,
    attempts: int = 0,
) -> tuple[Any, ...]:
    """Build one tuple in the order ``update_states`` expects."""
    return (file_path, inode, device, mtime, size, status, model_used, attempts)


def utc_now() -> str:
    """Return the current time as an ISO 8601 string in UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
