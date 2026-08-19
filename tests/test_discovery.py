"""Tests for media discovery."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aisrt.config import FilterConfig
from aisrt.discovery import (
    ACTION_PROCESS,
    MAX_ATTEMPTS,
    DiscoveryEngine,
    MediaFile,
    iter_media_files,
)
from aisrt.probing import AudioTrack, MediaInfo
from aisrt.state import STATUS_COMPLETED, STATUS_FAILED, SkipRecord, StateTracker

DEFAULT_EXTENSIONS = frozenset({".mkv", ".mp4"})


def _touch(path: Path, age_minutes: float = 60.0) -> Path:
    """Create a file and backdate it so the age filter lets it through."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")
    stamp = time.time() - age_minutes * 60
    os.utime(path, (stamp, stamp))
    return path


class TestWalker:
    """The walker must find every media file and skip everything else."""

    def test_finds_files_in_subdirectories(self, tmp_path: Path) -> None:
        """Nested directories are crawled."""
        _touch(tmp_path / "a.mkv")
        _touch(tmp_path / "shows" / "season 1" / "b.mp4")
        _touch(tmp_path / "notes.txt")

        found = {file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ())}
        assert found == {"a.mkv", "b.mp4"}

    def test_extension_matching_ignores_case(self, tmp_path: Path) -> None:
        """An uppercase extension is still a media file."""
        _touch(tmp_path / "movie.MKV")
        found = {file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ())}
        assert found == {"movie.MKV"}

    @pytest.mark.parametrize("directory", ["Sample", "sample", "SAMPLE", "Extras", "Featurettes"])
    def test_excludes_a_directory_whatever_its_case(self, tmp_path: Path, directory: str) -> None:
        """Real libraries capitalise these directories, so matching must not care."""
        _touch(tmp_path / "keep.mkv")
        _touch(tmp_path / directory / "skip.mkv")

        patterns = ("*sample*", "*extras*", "*featurettes*")
        found = {
            file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, patterns)
        }
        assert found == {"keep.mkv"}

    @pytest.mark.parametrize("filename", ["Movie.Sample.mkv", "movie-SAMPLE.mkv"])
    def test_excludes_a_file_whatever_its_case(self, tmp_path: Path, filename: str) -> None:
        """The same rule applies to a file name."""
        _touch(tmp_path / "keep.mkv")
        _touch(tmp_path / filename)

        found = {
            file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ("*sample*",))
        }
        assert found == {"keep.mkv"}

    def test_reads_file_metadata(self, tmp_path: Path) -> None:
        """Size, inode, and device come from a single stat call."""
        path = _touch(tmp_path / "movie.mkv")
        stat = path.stat()

        found = next(iter(iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ())))
        assert found.size == stat.st_size
        assert found.inode == stat.st_ino
        assert found.device == stat.st_dev

    def test_hidden_files_are_skipped(self, tmp_path: Path) -> None:
        """The tool's own temporary files start with a dot."""
        _touch(tmp_path / "keep.mkv")
        _touch(tmp_path / ".movie.abc123.srt.tmp")
        _touch(tmp_path / ".hidden.mkv")

        found = {file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ())}
        assert found == {"keep.mkv"}

    def test_an_unreadable_directory_does_not_stop_the_walk(self, tmp_path: Path) -> None:
        """One bad subtree must not lose the rest of the library."""
        _touch(tmp_path / "keep.mkv")
        blocked = tmp_path / "blocked"
        _touch(blocked / "hidden.mkv")
        os.chmod(blocked, 0o000)
        try:
            found = {file.path.name for file in iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ())}
            assert "keep.mkv" in found
        finally:
            os.chmod(blocked, 0o755)

    def test_symlinked_directories_are_skipped_by_default(self, tmp_path: Path) -> None:
        """Following symlinks is opt-in, because it risks a loop."""
        real = tmp_path / "real"
        _touch(real / "movie.mkv")
        (tmp_path / "link").symlink_to(real, target_is_directory=True)

        found = list(iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ()))
        assert len(found) == 1

    def test_a_symlink_loop_terminates(self, tmp_path: Path) -> None:
        """With following enabled, a cycle is detected instead of hanging."""
        nested = tmp_path / "a" / "b"
        _touch(nested / "movie.mkv")
        (nested / "loop").symlink_to(tmp_path / "a", target_is_directory=True)

        found = list(iter_media_files(tmp_path, DEFAULT_EXTENSIONS, (), follow_symlinks=True))
        assert len(found) == 1

    def test_a_deep_tree_does_not_exhaust_the_stack(self, tmp_path: Path) -> None:
        """The walk is iterative, so a deep tree cannot raise RecursionError."""
        # Single-character names keep the absolute path under the platform limit
        # while still nesting far deeper than a recursive walk would tolerate.
        original = os.getcwd()
        os.chdir(tmp_path)
        try:
            for _ in range(400):
                os.mkdir("d")
                os.chdir("d")
            Path("movie.mkv").write_text("data")
        finally:
            os.chdir(original)

        found = list(iter_media_files(tmp_path, DEFAULT_EXTENSIONS, ()))
        assert len(found) == 1

    def test_a_missing_root_yields_nothing(self, tmp_path: Path) -> None:
        """A path that does not exist is reported, not raised."""
        assert list(iter_media_files(tmp_path / "absent", DEFAULT_EXTENSIONS, ())) == []


def _media_info(with_subtitle: bool = False) -> MediaInfo:
    """Build a MediaInfo with one audio track."""
    return MediaInfo(
        duration=120.0,
        audio_tracks=(AudioTrack(0, "en", "aac", 2, True, None),),
        text_subtitle_languages=frozenset({"en"}) if with_subtitle else frozenset(),
    )


class TestAnalysis:
    """Each file gets exactly one action."""

    @pytest.fixture
    def tracker(self) -> AsyncMock:
        """Build a mocked state store with nothing recorded yet."""
        store = AsyncMock(spec=StateTracker)
        store.get_skip_records.return_value = {}
        store.get_all_processed_hardlinks.return_value = set()
        store.update_states = AsyncMock()
        return store

    @pytest.fixture
    def engine(self, tmp_path: Path, tracker: AsyncMock) -> DiscoveryEngine:
        """Build an engine over a temporary library with a mocked store."""
        config = FilterConfig(
            min_age_mins=15,
            extensions=[".mkv"],
            exclude_patterns=["*sample*"],
            target_languages=["en"],
        )
        return DiscoveryEngine(tmp_path, config, tracker)

    async def _scan(self, engine: DiscoveryEngine) -> dict[str, str]:
        """Run a scan and index the actions by file name."""
        return {file.path.name: action async for file, action in engine.scan()}

    @pytest.mark.asyncio
    async def test_a_file_needing_a_subtitle_is_processed(
        self, engine: DiscoveryEngine, tmp_path: Path
    ) -> None:
        """The happy path reaches the pipeline."""
        _touch(tmp_path / "movie.mkv")
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert actions["movie.mkv"] == ACTION_PROCESS

    @pytest.mark.asyncio
    async def test_a_recent_file_is_skipped(self, engine: DiscoveryEngine, tmp_path: Path) -> None:
        """An active download must not be parsed."""
        _touch(tmp_path / "downloading.mkv", age_minutes=1)
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert "Modified recently" in actions["downloading.mkv"]

    @pytest.mark.asyncio
    async def test_a_sidecar_subtitle_is_skipped(
        self, engine: DiscoveryEngine, tmp_path: Path
    ) -> None:
        """An existing .srt next to the video ends the work."""
        _touch(tmp_path / "movie.mkv")
        _touch(tmp_path / "movie.en.srt")
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert "External sibling subtitle" in actions["movie.mkv"]

    @pytest.mark.asyncio
    async def test_an_embedded_subtitle_is_skipped_and_recorded(
        self, engine: DiscoveryEngine, tmp_path: Path, tracker: AsyncMock
    ) -> None:
        """A text subtitle inside the container is recorded so it is found faster next time."""
        _touch(tmp_path / "movie.mkv")
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info(True))):
            actions = await self._scan(engine)

        assert "Embedded subtitle detected" in actions["movie.mkv"]
        tracker.update_states.assert_awaited()

    @pytest.mark.asyncio
    async def test_a_completed_file_is_skipped(
        self, engine: DiscoveryEngine, tmp_path: Path, tracker: AsyncMock
    ) -> None:
        """A finished row of the same size stops the work."""
        path = _touch(tmp_path / "movie.mkv")
        tracker.get_skip_records.return_value = {
            str(path): SkipRecord(STATUS_COMPLETED, path.stat().st_size, path.stat().st_mtime, 0)
        }
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert "Already processed" in actions["movie.mkv"]

    @pytest.mark.asyncio
    async def test_a_resized_file_is_reprocessed(
        self, engine: DiscoveryEngine, tmp_path: Path, tracker: AsyncMock
    ) -> None:
        """A re-encode changes the size, so the old result no longer applies."""
        path = _touch(tmp_path / "movie.mkv")
        tracker.get_skip_records.return_value = {
            str(path): SkipRecord(STATUS_COMPLETED, 999_999, path.stat().st_mtime, 0)
        }
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert actions["movie.mkv"] == ACTION_PROCESS

    @pytest.mark.asyncio
    async def test_a_repeatedly_failing_file_is_given_up_on(
        self, engine: DiscoveryEngine, tmp_path: Path, tracker: AsyncMock
    ) -> None:
        """A file that always fails stops consuming GPU time every run."""
        path = _touch(tmp_path / "broken.mkv")
        tracker.get_skip_records.return_value = {
            str(path): SkipRecord(
                STATUS_FAILED, path.stat().st_size, path.stat().st_mtime, MAX_ATTEMPTS
            )
        }
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert "Failed" in actions["broken.mkv"]

    @pytest.mark.asyncio
    async def test_a_hardlink_is_skipped(
        self, engine: DiscoveryEngine, tmp_path: Path, tracker: AsyncMock
    ) -> None:
        """The same content under a second path is not transcribed twice."""
        path = _touch(tmp_path / "movie.mkv")
        stat = path.stat()
        tracker.get_all_processed_hardlinks.return_value = {
            (stat.st_dev, stat.st_ino, stat.st_size)
        }
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            actions = await self._scan(engine)
        assert "Hardlink" in actions["movie.mkv"]

    @pytest.mark.asyncio
    async def test_a_file_without_audio_is_skipped(
        self, engine: DiscoveryEngine, tmp_path: Path
    ) -> None:
        """There is nothing to transcribe without an audio track."""
        _touch(tmp_path / "silent.mkv")
        info = MediaInfo(duration=120.0, audio_tracks=())
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=info)):
            actions = await self._scan(engine)
        assert "No audio track" in actions["silent.mkv"]

    @pytest.mark.asyncio
    async def test_an_unreadable_file_is_skipped(
        self, engine: DiscoveryEngine, tmp_path: Path
    ) -> None:
        """A failed probe must not be mistaken for "no subtitles"."""
        _touch(tmp_path / "corrupt.mkv")
        info = MediaInfo(duration=None, probe_failed=True)
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=info)):
            actions = await self._scan(engine)
        assert "Cannot read" in actions["corrupt.mkv"]

    @pytest.mark.asyncio
    async def test_the_probe_result_travels_with_the_file(
        self, engine: DiscoveryEngine, tmp_path: Path
    ) -> None:
        """The pipeline reuses the duration and track list, so it never reprobes."""
        _touch(tmp_path / "movie.mkv")
        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=_media_info())):
            files = [file async for file, _ in engine.scan()]

        assert files[0].media_info is not None
        assert files[0].duration == 120.0

    @pytest.mark.asyncio
    async def test_probing_is_bounded(self, engine: DiscoveryEngine, tmp_path: Path) -> None:
        """Concurrent probes never exceed the configured limit."""
        for index in range(20):
            _touch(tmp_path / f"movie{index}.mkv")

        live = 0
        peak = 0

        async def counting_probe(_path: Path, timeout: float = 60.0) -> MediaInfo:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            try:
                return _media_info()
            finally:
                live -= 1

        with patch("aisrt.discovery.probe_media", new=counting_probe):
            actions = await self._scan(engine)

        assert len(actions) == 20
        assert peak <= 4


class TestMediaFile:
    """The dataclass exposes what the pipeline needs."""

    def test_duration_is_none_without_a_probe(self) -> None:
        """A file that was never probed reports no duration."""
        assert MediaFile(Path("a.mkv"), 1, 1.0, 1).duration is None

    def test_duration_comes_from_the_probe(self) -> None:
        """A probed file reports the container duration."""
        file = MediaFile(Path("a.mkv"), 1, 1.0, 1, media_info=_media_info())
        assert file.duration == 120.0
