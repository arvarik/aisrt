"""NAS-safe media discovery."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import time
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Final

from loguru import logger

from aisrt.config import FilterConfig
from aisrt.probing import MediaInfo, external_subtitle_path, normalize_languages, probe_media
from aisrt.state import (
    STATUS_EMBEDDED_EXISTS,
    STATUS_FAILED,
    SkipRecord,
    StateTracker,
    build_row,
)

ACTION_PROCESS: Final = "PROCESS"
"""The action string that means the file needs a subtitle."""

_BATCH_SIZE: Final = 256
_STATE_FLUSH_SIZE: Final = 200
MAX_ATTEMPTS: Final = 3
"""How many times a failing file is retried across runs before it is left alone."""


@dataclass(slots=True)
class MediaFile:
    """A media file found on disk."""

    path: Path
    size: int
    mtime: float
    inode: int
    device: int = 0
    media_info: MediaInfo | None = None
    external_subtitle: Path | None = None

    @property
    def duration(self) -> float | None:
        """The probed duration in seconds, or None when it is unknown."""
        return self.media_info.duration if self.media_info else None


def iter_media_files(
    root: Path,
    extensions: frozenset[str],
    exclude_patterns: tuple[str, ...],
    follow_symlinks: bool = False,
) -> Iterator[MediaFile]:
    """Walk a directory tree and yield media files as they are found.

    The walk uses an explicit stack, so a deep tree cannot exhaust the recursion
    limit. It costs one ``lstat`` per media file and none for anything else. A
    failure on one entry or one directory never stops the walk.

    Args:
        root: The directory to crawl.
        extensions: Lowercase extensions to accept, each with a leading dot.
        exclude_patterns: Lowercase glob patterns. A match on any path component
            skips that file, or prunes that whole subtree.
        follow_symlinks: If True, descend into symlinked directories. Loops are
            detected by device and inode.

    Yields:
        One MediaFile per accepted file.
    """
    stack: list[str] = [str(root)]
    visited: set[tuple[int, int]] = set()

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    name = entry.name.lower()
                    if name.startswith("."):
                        continue
                    if any(fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            if follow_symlinks:
                                stat = entry.stat()
                                key = (stat.st_dev, stat.st_ino)
                                if key in visited:
                                    continue
                                visited.add(key)
                            stack.append(entry.path)
                        elif os.path.splitext(name)[1] in extensions and entry.is_file():
                            # is_file() follows the link and rejects a FIFO, a
                            # socket, and a broken symlink. stat() then describes
                            # the target, so size and inode identify real content.
                            stat = entry.stat()
                            yield MediaFile(
                                path=Path(entry.path),
                                size=stat.st_size,
                                mtime=stat.st_mtime,
                                inode=stat.st_ino,
                                device=stat.st_dev,
                            )
                    except OSError as error:
                        logger.debug(f"Skipping {entry.path}: {error}")
        except OSError as error:
            logger.warning(f"Cannot read directory {current}: {error}")


class DiscoveryEngine:
    """Crawls a media directory and decides what each file needs."""

    def __init__(
        self,
        media_dir: Path,
        config: FilterConfig,
        state_tracker: StateTracker,
        probe_concurrency: int = 4,
    ) -> None:
        """Initialize the engine.

        Args:
            media_dir: The directory to crawl.
            config: The filtering rules.
            state_tracker: The open state database.
            probe_concurrency: How many ffprobe calls may run at once.
        """
        self.media_dir = media_dir
        self.config = config
        self.state_tracker = state_tracker
        self.target_languages = normalize_languages(config.target_languages)
        self.extensions = frozenset(config.extensions)
        self.exclude_patterns = tuple(config.exclude_patterns)
        self.probe_concurrency = max(1, probe_concurrency)
        self._probe_semaphore = asyncio.Semaphore(self.probe_concurrency)
        # The pools are created per scan, so one engine can be scanned again.
        self._crawl_pool: ThreadPoolExecutor | None = None
        self._stat_pool: ThreadPoolExecutor | None = None

    async def scan(self) -> AsyncGenerator[tuple[MediaFile, str], None]:
        """Crawl the directory and report what to do with each file.

        Files stream out as the crawl finds them, so the pipeline starts working
        before the crawl finishes. Probing runs concurrently, bounded by the
        semaphore, which is what makes a large library scan quickly.

        Yields:
            A tuple of the file and an action string. The action is ``PROCESS``
            or a string that starts with ``SKIP: ``.
        """
        logger.info(f"Scanning {self.media_dir}")
        # The walk drives one generator, so it needs a single thread. The sidecar
        # checks get their own pool, so a batch of stat calls cannot hold the
        # crawl back.
        self._crawl_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aisrt-crawl")
        self._stat_pool = ThreadPoolExecutor(
            max_workers=self.probe_concurrency, thread_name_prefix="aisrt-stat"
        )
        skip_records = await self.state_tracker.get_skip_records(str(self.media_dir))
        processed_identities = await self.state_tracker.get_all_processed_hardlinks()
        now = time.time()

        walker = iter_media_files(
            self.media_dir, self.extensions, self.exclude_patterns, self.config.follow_symlinks
        )
        loop = asyncio.get_running_loop()
        found = 0
        pending_rows: list[dict[str, Any]] = []

        try:
            while True:
                batch = await loop.run_in_executor(
                    self._crawl_pool, _take_batch, walker, _BATCH_SIZE
                )
                if not batch:
                    break
                found += len(batch)

                results = await asyncio.gather(
                    *(
                        self._analyze_file(media_file, now, skip_records, processed_identities)
                        for media_file in batch
                    )
                )
                for media_file, action in results:
                    if action.startswith("SKIP: Embedded"):
                        pending_rows.append(
                            build_row(
                                str(media_file.path),
                                media_file.inode,
                                media_file.device,
                                media_file.mtime,
                                media_file.size,
                                STATUS_EMBEDDED_EXISTS,
                            )
                        )
                        processed_identities.add(
                            (media_file.device, media_file.inode, media_file.size)
                        )
                    if len(pending_rows) >= _STATE_FLUSH_SIZE:
                        await self.state_tracker.update_states(pending_rows)
                        pending_rows.clear()
                    yield media_file, action
        finally:
            if pending_rows:
                await self.state_tracker.update_states(pending_rows)
            self._crawl_pool.shutdown(wait=False)
            self._stat_pool.shutdown(wait=False)
            self._crawl_pool = None
            self._stat_pool = None
            logger.info(f"Scan finished. {found} media file(s) examined.")

    async def _analyze_file(
        self,
        media_file: MediaFile,
        now: float,
        skip_records: dict[str, SkipRecord],
        processed_identities: set[tuple[int, int, int]],
    ) -> tuple[MediaFile, str]:
        """Decide what one file needs.

        Args:
            media_file: The file to inspect.
            now: The time the scan started, in seconds since the epoch.
            skip_records: The state rows loaded once for the whole scan.
            processed_identities: Content identities already finished.

        Returns:
            The file and its action string.
        """
        age_seconds = now - media_file.mtime
        if age_seconds < self.config.min_age_mins * 60:
            return media_file, f"SKIP: Modified recently (< {self.config.min_age_mins}m)"

        loop = asyncio.get_running_loop()
        sidecar = await loop.run_in_executor(
            self._stat_pool, external_subtitle_path, media_file.path, self.target_languages
        )
        if sidecar is not None:
            media_file.external_subtitle = sidecar
            return media_file, "SKIP: External sibling subtitle exists"

        record = skip_records.get(str(media_file.path))
        if record is not None:
            if record.status in {"COMPLETED", "NO_SPEECH"} and record.size == media_file.size:
                return media_file, "SKIP: Already processed (database)"
            if record.status == STATUS_EMBEDDED_EXISTS and record.size == media_file.size:
                return media_file, "SKIP: Embedded subtitle recorded (database)"
            if record.status == STATUS_FAILED and record.attempts >= MAX_ATTEMPTS:
                return media_file, f"SKIP: Failed {record.attempts} times"

        identity = (media_file.device, media_file.inode, media_file.size)
        if identity in processed_identities:
            return media_file, "SKIP: Hardlink to an already processed file"

        async with self._probe_semaphore:
            media_file.media_info = await probe_media(media_file.path)

        info = media_file.media_info
        if info.probe_failed:
            return media_file, "SKIP: Cannot read the media file"
        if not info.audio_tracks:
            return media_file, "SKIP: No audio track"
        if info.has_text_subtitle(self.target_languages):
            return media_file, "SKIP: Embedded subtitle detected"

        return media_file, ACTION_PROCESS


def _take_batch(iterator: Iterator[MediaFile], size: int) -> list[MediaFile]:
    """Pull up to ``size`` items from an iterator. Runs on the crawl thread."""
    return list(islice(iterator, size))
