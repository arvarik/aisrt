"""Tests for the concurrent pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aisrt.config import FilterConfig
from aisrt.discovery import ACTION_PROCESS, DiscoveryEngine, MediaFile
from aisrt.pipeline import InferenceJob, MemoryBudget, Pipeline, PipelineStats
from aisrt.probing import AudioTrack, MediaInfo
from aisrt.state import (
    STATUS_COMPLETED,
    STATUS_EXTRACTING,
    STATUS_INFERENCING,
    STATUS_NO_SPEECH,
    StateTracker,
)

SRT_BLOCK = "1\n00:00:00,000 --> 00:00:02,000\nHello.\n"


def media_file(name: str, duration: float = 60.0) -> MediaFile:
    """Build a discovered file with a probe result attached."""
    return MediaFile(
        path=Path(f"/media/{name}"),
        size=1024,
        mtime=1.0,
        inode=abs(hash(name)) % 100_000,
        device=1,
        media_info=MediaInfo(
            duration=duration, audio_tracks=(AudioTrack(0, "en", "aac", 2, True, None),)
        ),
    )


@pytest.fixture
def tracker() -> AsyncMock:
    """Build a mocked state store that records every transition."""
    store = AsyncMock(spec=StateTracker)
    store.get_state.return_value = None
    return store


@pytest.fixture
def engine(tracker: AsyncMock) -> DiscoveryEngine:
    """Build a discovery engine that yields two files to process and one to skip."""
    fake = MagicMock(spec=DiscoveryEngine)
    fake.state_tracker = tracker
    fake.config = FilterConfig(target_languages=["en"])
    fake.target_languages = frozenset({"en"})

    async def scan() -> AsyncIterator[tuple[MediaFile, str]]:
        yield media_file("one.mkv"), ACTION_PROCESS
        yield media_file("two.mkv"), ACTION_PROCESS
        yield media_file("three.mkv"), "SKIP: Already processed (database)"

    fake.scan = scan
    return fake


@pytest.fixture
def stt() -> MagicMock:
    """Build a worker that returns one subtitle block for any audio."""
    worker = MagicMock()
    worker.model_name = "tiny.en"
    worker.detect_language.return_value = ("en", 0.99)
    worker.transcribe.return_value = (iter([]), 60.0)
    worker.executor = None
    return worker


class TestMemoryBudget:
    """Memory must be bounded in bytes, not in file count."""

    @pytest.mark.asyncio
    async def test_reservations_add_up(self) -> None:
        """Two small reservations both fit."""
        budget = MemoryBudget(100 * 1024 * 1024)
        await budget.acquire(10 * 1024 * 1024)
        await budget.acquire(10 * 1024 * 1024)
        assert budget._used == 20 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_a_waiter_resumes_after_a_release(self) -> None:
        """The budget blocks until room appears, rather than overcommitting."""
        budget = MemoryBudget(100 * 1024 * 1024)
        await budget.acquire(90 * 1024 * 1024)

        waiter = asyncio.create_task(budget.acquire(50 * 1024 * 1024))
        await asyncio.sleep(0)
        assert not waiter.done()

        await budget.release(90 * 1024 * 1024)
        await asyncio.wait_for(waiter, timeout=1.0)

    @pytest.mark.asyncio
    async def test_a_file_larger_than_the_budget_still_runs(self) -> None:
        """One oversized file must not deadlock the pipeline forever."""
        budget = MemoryBudget(10 * 1024 * 1024)
        await asyncio.wait_for(budget.acquire(500 * 1024 * 1024), timeout=1.0)

    @pytest.mark.asyncio
    async def test_releasing_more_than_reserved_is_safe(self) -> None:
        """The counter never goes negative."""
        budget = MemoryBudget(100 * 1024 * 1024)
        await budget.release(999)
        assert budget._used == 0

    @pytest.mark.asyncio
    async def test_adjust_never_waits(self) -> None:
        """Correcting a reservation upward must not block.

        The memory already exists. Waiting for room a holder is itself occupying
        deadlocks the moment every worker under-estimates at once.
        """
        budget = MemoryBudget(100 * 1024 * 1024)
        await budget.acquire(100 * 1024 * 1024)
        await asyncio.wait_for(budget.adjust(50 * 1024 * 1024), timeout=1.0)
        assert budget._used == 150 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_every_worker_underestimating_does_not_deadlock(self) -> None:
        """Three workers that each need a little more than they reserved finish."""
        budget = MemoryBudget(300)

        async def worker() -> None:
            await budget.acquire(100)
            await asyncio.sleep(0)
            await budget.adjust(20)
            await budget.release(120)

        await asyncio.wait_for(asyncio.gather(*(worker() for _ in range(3))), timeout=2.0)
        assert budget._used == 0

    @pytest.mark.asyncio
    async def test_adjust_downward_wakes_a_waiter(self) -> None:
        """Freeing part of a reservation lets a blocked worker through."""
        megabyte = 1024 * 1024
        budget = MemoryBudget(100 * megabyte)
        await budget.acquire(100 * megabyte)

        waiter = asyncio.create_task(budget.acquire(60 * megabyte))
        await asyncio.sleep(0)
        assert not waiter.done(), "the waiter should block while the budget is full"

        # Dropping to 30 MB in use leaves room for the 60 MB request.
        await budget.adjust(-70 * megabyte)
        await asyncio.wait_for(waiter, timeout=1.0)
        assert budget._used == 90 * megabyte

    def test_the_limit_has_a_sane_floor(self) -> None:
        """A tiny limit is raised, so one minute of audio always fits."""
        assert MemoryBudget(1).limit_bytes >= 16_000 * 4 * 60


class TestPipelineStats:
    """The summary numbers must be honest."""

    def test_speedup_needs_elapsed_time(self) -> None:
        """A run that took no time reports no speedup instead of dividing by zero."""
        stats = PipelineStats(start_time=5.0, end_time=5.0, total_audio_duration_secs=100.0)
        assert stats.speedup == 0.0

    def test_speedup(self) -> None:
        """Ten minutes of audio in one minute is ten times real time."""
        stats = PipelineStats(start_time=0.0, end_time=60.0, total_audio_duration_secs=600.0)
        assert stats.speedup == pytest.approx(10.0)

    def test_failures_are_reported(self) -> None:
        """The flag drives the process exit code."""
        assert PipelineStats(files_failed=1).had_failures is True
        assert PipelineStats(files_failed=0).had_failures is False


class TestPipelineRun:
    """A full pass must transcribe every file that needs it."""

    @pytest.mark.asyncio
    async def test_processes_each_file_and_writes_a_subtitle(
        self, engine: DiscoveryEngine, stt: MagicMock
    ) -> None:
        """Two files are transcribed, one is skipped, and both subtitles land."""
        pipeline = Pipeline(engine, stt)
        written: list[tuple[Path, str, str]] = []

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch(
                "aisrt.pipeline.AtomicWriter.write_srt",
                side_effect=lambda p, c, lang: written.append((p, c, lang)),
            ),
        ):
            stats = await pipeline.run()

        assert stats.files_scanned == 3
        assert stats.files_skipped == 1
        assert stats.files_processed == 2
        assert stats.files_failed == 0
        assert stats.total_audio_duration_secs == pytest.approx(120.0)
        assert {path.name for path, _, _ in written} == {"one.mkv", "two.mkv"}
        assert {lang for _, _, lang in written} == {"en"}

    @pytest.mark.asyncio
    async def test_state_transitions_are_recorded(
        self, engine: DiscoveryEngine, stt: MagicMock, tracker: AsyncMock
    ) -> None:
        """Every file walks through extracting, inferencing, then completed."""
        pipeline = Pipeline(engine, stt)

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt"),
        ):
            await pipeline.run()

        statuses = [call.kwargs["status"] for call in tracker.update_state.await_args_list]
        assert STATUS_EXTRACTING in statuses
        assert STATUS_INFERENCING in statuses
        assert statuses.count(STATUS_COMPLETED) == 2

    @pytest.mark.asyncio
    async def test_the_model_name_is_recorded(
        self, engine: DiscoveryEngine, stt: MagicMock, tracker: AsyncMock
    ) -> None:
        """The database says which model produced the subtitle, not 'unknown'."""
        pipeline = Pipeline(engine, stt)

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt"),
        ):
            await pipeline.run()

        models = {
            call.kwargs.get("model_used")
            for call in tracker.update_state.await_args_list
            if call.kwargs["status"] == STATUS_COMPLETED
        }
        assert models == {"tiny.en"}

    @pytest.mark.asyncio
    async def test_a_bad_file_does_not_stop_the_run(
        self, engine: DiscoveryEngine, stt: MagicMock
    ) -> None:
        """One corrupt file is marked failed and the pipeline carries on."""
        pipeline = Pipeline(engine, stt)
        calls = {"count": 0}

        async def flaky(*_args: object, **_kwargs: object) -> np.ndarray:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("FFmpeg extraction failed")
            return np.zeros(16000, dtype=np.float32)

        with (
            patch("aisrt.pipeline.AudioExtractor.extract_audio_to_memory", new=flaky),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt"),
        ):
            stats = await pipeline.run()

        assert stats.files_failed == 1
        assert stats.files_processed == 1

    @pytest.mark.asyncio
    async def test_silence_is_final_not_a_failure(
        self, engine: DiscoveryEngine, stt: MagicMock, tracker: AsyncMock
    ) -> None:
        """Recording no-speech stops the file being retranscribed every run."""
        pipeline = Pipeline(engine, stt)

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=("", 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt") as writer,
        ):
            stats = await pipeline.run()

        assert stats.files_without_speech == 2
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        writer.assert_not_called()

        statuses = [call.kwargs["status"] for call in tracker.update_state.await_args_list]
        assert statuses.count(STATUS_NO_SPEECH) == 2

    @pytest.mark.asyncio
    async def test_a_database_error_does_not_abort_the_run(
        self, engine: DiscoveryEngine, stt: MagicMock, tracker: AsyncMock
    ) -> None:
        """A locked database must not take the whole pipeline down."""
        tracker.update_state.side_effect = RuntimeError("database is locked")
        pipeline = Pipeline(engine, stt)

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt"),
        ):
            stats = await pipeline.run()

        assert stats.files_processed == 2

    @pytest.mark.asyncio
    async def test_a_stop_request_drains_instead_of_dropping_work(
        self, engine: DiscoveryEngine, stt: MagicMock
    ) -> None:
        """A shutdown finishes what is in flight and stops queueing more."""
        stop = asyncio.Event()
        stop.set()
        pipeline = Pipeline(engine, stt, stop_event=stop)

        with (
            patch(
                "aisrt.pipeline.AudioExtractor.extract_audio_to_memory",
                new=AsyncMock(return_value=np.zeros(16000, dtype=np.float32)),
            ),
            patch.object(Pipeline, "_run_inference", return_value=(SRT_BLOCK, 60.0)),
            patch("aisrt.pipeline.AtomicWriter.write_srt"),
        ):
            stats = await asyncio.wait_for(pipeline.run(), timeout=5.0)

        # The crawl stops as soon as it meets work it must not start, so the
        # run ends promptly instead of walking the rest of the library.
        assert stats.files_processed == 0
        assert stats.files_scanned < 3

    @pytest.mark.asyncio
    async def test_an_empty_library_finishes_cleanly(self, stt: MagicMock) -> None:
        """A directory with nothing to do must not hang on a sentinel."""
        tracker = AsyncMock(spec=StateTracker)
        empty = MagicMock(spec=DiscoveryEngine)
        empty.state_tracker = tracker
        empty.config = FilterConfig(target_languages=["en"])
        empty.target_languages = frozenset({"en"})

        async def scan() -> AsyncIterator[tuple[MediaFile, str]]:
            return
            yield  # pragma: no cover

        empty.scan = scan
        stats = await asyncio.wait_for(Pipeline(empty, stt).run(), timeout=5.0)
        assert stats.files_scanned == 0


class TestAudioTrackChoice:
    """Translating needs the original audio, not the English dub."""

    @pytest.mark.asyncio
    async def test_transcribing_prefers_the_target_language(
        self, engine: DiscoveryEngine, stt: MagicMock
    ) -> None:
        """A transcribe run wants the English track when one exists."""
        pipeline = Pipeline(engine, stt, translate=False)
        assert pipeline.preferred_audio == frozenset({"en"})

    @pytest.mark.asyncio
    async def test_translating_keeps_the_container_order(
        self, engine: DiscoveryEngine, stt: MagicMock
    ) -> None:
        """A translate run must not select the English track and translate nothing."""
        pipeline = Pipeline(engine, stt, translate=True)
        assert pipeline.preferred_audio == frozenset()


class TestInferenceJob:
    """The job reports how much memory it holds."""

    def test_nbytes(self) -> None:
        """Float32 samples are four bytes each."""
        job = InferenceJob(media_file("a.mkv"), np.zeros(1000, dtype=np.float32))
        assert job.nbytes == 4000
