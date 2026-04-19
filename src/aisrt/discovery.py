"""NAS-Safe File Discovery Engine."""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from aisrt.config import FilterConfig
from aisrt.probing import has_embedded_subtitles, has_external_subtitle
from aisrt.state import StateTracker


@dataclass
class MediaFile:
    """Represents a discovered media file pending processing."""

    path: Path
    size: int
    mtime: float
    inode: int


class DiscoveryEngine:
    """Safely crawls a media directory and filters files based on state and config."""

    def __init__(self, media_dir: Path, config: FilterConfig, state_tracker: StateTracker) -> None:
        """Initialize the discovery engine.

        Args:
            media_dir: The root directory to scan.
            config: The filtering configuration rules.
            state_tracker: The active SQLite state tracker.
        """
        self.media_dir = media_dir
        self.config = config
        self.state_tracker = state_tracker

    async def scan(self) -> AsyncGenerator[tuple[MediaFile, str], None]:
        """Scan the media directory and yield files with their action status.

        Yields:
            A tuple of (MediaFile, action_string).
            action_string is 'PROCESS' if the file needs STT, or a 'SKIP: <reason>' string.
        """
        loop = asyncio.get_running_loop()

        def _walk(directory: Path) -> list[Path]:
            paths = []
            try:
                for entry in os.scandir(directory):
                    path = Path(entry.path)

                    if any(path.match(p) for p in self.config.exclude_patterns):
                        continue

                    if entry.is_dir(follow_symlinks=False):
                        paths.extend(_walk(path))
                    elif entry.is_file(follow_symlinks=False):
                        if path.suffix.lower() in self.config.extensions:
                            paths.append(path)
            except PermissionError:
                logger.warning(f"Permission denied: {directory}")
            return paths

        logger.info(f"Starting directory scan at {self.media_dir}...")
        all_files = await loop.run_in_executor(None, _walk, self.media_dir)
        logger.info(f"Found {len(all_files)} potential media files. Analyzing...")

        current_time = time.time()

        for file_path in all_files:
            try:
                stat = file_path.stat()
                media_file = MediaFile(
                    path=file_path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    inode=stat.st_ino,
                )
            except OSError as e:
                logger.warning(f"Could not stat {file_path}: {e}")
                continue

            action_str = await self._analyze_file(media_file, current_time)
            yield media_file, action_str

    async def _analyze_file(self, media_file: MediaFile, current_time: float) -> str:
        """Determine if a single file should be processed or skipped."""
        min_age_seconds = self.config.min_age_mins * 60

        if (current_time - media_file.mtime) < min_age_seconds:
            return f"SKIP: Modified recently (< {self.config.min_age_mins}m)"

        if has_external_subtitle(media_file.path, self.config.target_languages):
            return "SKIP: External sibling subtitle exists"

        db_state = await self.state_tracker.get_state(str(media_file.path))
        if db_state and db_state.status == "COMPLETED" and db_state.size == media_file.size:
            return "SKIP: Already processed (Database)"

        is_hardlink = await self.state_tracker.check_hardlink_processed(
            media_file.inode, media_file.size
        )
        if is_hardlink:
            return "SKIP: Hardlink to already processed file"

        if db_state and db_state.status == "EMBEDDED_EXISTS":
            return "SKIP: Embedded English subtitle exists (Database)"

        has_embedded = await has_embedded_subtitles(media_file.path, self.config.target_languages)
        if has_embedded:
            await self.state_tracker.update_state(
                file_path=str(media_file.path),
                inode=media_file.inode,
                mtime=media_file.mtime,
                size=media_file.size,
                status="EMBEDDED_EXISTS",
            )
            return "SKIP: Embedded English subtitle detected"

        return "PROCESS"
