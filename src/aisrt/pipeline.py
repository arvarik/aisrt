"""Bounded producer and consumer pipeline built on asyncio.TaskGroup."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from loguru import logger

from aisrt.assembly import AtomicWriter, SRTFormatter
from aisrt.discovery import ACTION_PROCESS, DiscoveryEngine, MediaFile
from aisrt.extractor import SAMPLE_RATE, AudioExtractor
from aisrt.state import (
    STATUS_COMPLETED,
    STATUS_EXTRACTING,
    STATUS_FAILED,
    STATUS_INFERENCING,
    STATUS_NO_SPEECH,
)
from aisrt.stt import STTWorker

_MAX_EXTRACTORS: Final = 3
"""Extraction is I/O bound and inference is the bottleneck, so a small pool is
enough to keep the model fed."""

_BYTES_PER_SAMPLE: Final = 4
_MIN_LANGUAGE_CONFIDENCE: Final = 0.5


@dataclass(slots=True)
class PipelineStats:
    """Counters for one pipeline run."""

    files_scanned: int = 0
    files_skipped: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_without_speech: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    total_audio_duration_secs: float = 0.0

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds the run took."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def speedup(self) -> float:
        """Audio seconds transcribed per second of wall-clock time."""
        return self.total_audio_duration_secs / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def had_failures(self) -> bool:
        """Whether any file failed."""
        return self.files_failed > 0


@dataclass(slots=True)
class InferenceJob:
    """One media file together with its decoded audio."""

    media_file: MediaFile
    audio_data: np.ndarray

    @property
    def nbytes(self) -> int:
        """Bytes of memory the decoded audio occupies."""
        return int(self.audio_data.nbytes)


class MemoryBudget:
    """Caps the decoded audio held in memory, counted in bytes.

    Bounding by file count is not enough. Three minutes of audio and three hours
    of audio occupy the same slot but differ by a factor of sixty in memory.
    """

    def __init__(self, limit_bytes: int) -> None:
        """Initialize the budget.

        Args:
            limit_bytes: The most decoded audio that may be resident at once.
        """
        self.limit_bytes = max(limit_bytes, SAMPLE_RATE * _BYTES_PER_SAMPLE * 60)
        self._used = 0
        self._condition = asyncio.Condition()

    async def acquire(self, size: int) -> None:
        """Wait until ``size`` bytes fit, then reserve them.

        A single file larger than the whole budget is admitted on its own rather
        than deadlocking the pipeline.
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._used == 0 or self._used + size <= self.limit_bytes
            )
            self._used += size

    async def release(self, size: int) -> None:
        """Return ``size`` bytes to the budget and wake a waiter."""
        async with self._condition:
            self._used = max(0, self._used - size)
            self._condition.notify_all()

    async def adjust(self, delta: int) -> None:
        """Correct a reservation to the size that was actually allocated.

        This never waits. The memory already exists, so blocking here would ask
        a holder to wait for room it is itself occupying, which deadlocks as soon
        as every worker under-estimates at the same time. Going slightly over the
        limit is the honest outcome; the next acquire absorbs it.
        """
        if delta == 0:
            return
        async with self._condition:
            self._used = max(0, self._used + delta)
            if delta < 0:
                self._condition.notify_all()


class Pipeline:
    """Runs discovery, extraction, and inference as bounded concurrent stages."""

    def __init__(
        self,
        discovery_engine: DiscoveryEngine,
        stt_worker: STTWorker,
        formatter: SRTFormatter | None = None,
        translate: bool = False,
        language: str | None = None,
        max_memory_mb: int = 2048,
        extract_timeout_secs: float = 1800.0,
        stop_event: asyncio.Event | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            discovery_engine: The crawler that decides which files to process.
            stt_worker: The loaded model and its worker thread.
            formatter: The subtitle formatter. A default one is built when None.
            translate: True to translate speech into English.
            language: Force the spoken language. Detect it when None.
            max_memory_mb: Cap on decoded audio held in memory.
            extract_timeout_secs: Seconds to wait for FFmpeg on one file.
            stop_event: Set this to stop the producer and drain what is in flight.
            progress: Receives per-file progress updates on the event loop.
        """
        self.discovery = discovery_engine
        self.stt = stt_worker
        self.formatter = formatter or SRTFormatter()
        self.translate = translate
        self.language = language
        self.extract_timeout_secs = extract_timeout_secs
        self.stop_event = stop_event or asyncio.Event()
        self.progress = progress
        self.state = discovery_engine.state_tracker
        self.subtitle_language = discovery_engine.config.target_languages[0]
        self.preferred_audio = frozenset() if translate else discovery_engine.target_languages

        self.extractor_count = _MAX_EXTRACTORS
        self.extraction_queue: asyncio.Queue[MediaFile | None] = asyncio.Queue(
            maxsize=self.extractor_count * 4
        )
        self.inference_queue: asyncio.Queue[InferenceJob | None] = asyncio.Queue(maxsize=2)
        self.budget = MemoryBudget(max_memory_mb * 1024 * 1024)
        self.stats = PipelineStats()

    async def run(self) -> PipelineStats:
        """Run one full pass over the library and return the counters."""
        logger.debug("Pipeline starting.")
        self.stats.start_time = time.time()
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._orchestrate(group))
        finally:
            self.stats.end_time = time.time()
        return self.stats

    async def _orchestrate(self, group: asyncio.TaskGroup) -> None:
        """Start the workers and shut them down in order."""
        producer = group.create_task(self._produce(), name="producer")
        extractors = [
            group.create_task(self._extract(index), name=f"extractor-{index}")
            for index in range(self.extractor_count)
        ]
        inference = group.create_task(self._infer(), name="inference")

        await producer
        for extractor in extractors:
            await extractor
        await self.inference_queue.put(None)
        await inference

    async def _produce(self) -> None:
        """Crawl the library and queue the files that need a subtitle."""
        try:
            async for media_file, action in self.discovery.scan():
                self.stats.files_scanned += 1
                if action != ACTION_PROCESS:
                    self.stats.files_skipped += 1
                    continue
                if self.stop_event.is_set():
                    logger.info("Shutdown requested. The producer stops queueing new files.")
                    break
                await self.extraction_queue.put(media_file)
        finally:
            # Every extractor takes exactly one sentinel and then exits.
            for _ in range(self.extractor_count):
                await self.extraction_queue.put(None)

    async def _extract(self, worker_id: int) -> None:
        """Decode audio into memory and hand it to the inference worker."""
        while True:
            media_file = await self.extraction_queue.get()
            if media_file is None:
                break

            reserved = 0
            try:
                await self._set_status(media_file, STATUS_EXTRACTING)

                info = media_file.media_info
                track_index = info.select_audio_track(self.preferred_audio) if info else 0
                estimated = _estimate_bytes(media_file.duration)
                await self.budget.acquire(estimated)
                reserved = estimated

                audio = await AudioExtractor.extract_audio_to_memory(
                    media_file.path,
                    track_index,
                    timeout=self.extract_timeout_secs,
                    duration=media_file.duration,
                )
                actual = int(audio.nbytes)
                await self.budget.adjust(actual - reserved)
                reserved = actual

                await self.inference_queue.put(InferenceJob(media_file, audio))
                reserved = 0
            except asyncio.CancelledError:
                if reserved:
                    await asyncio.shield(self.budget.release(reserved))
                raise
            except Exception as error:
                if reserved:
                    await self.budget.release(reserved)
                logger.error(f"Extractor {worker_id} failed on {media_file.path.name}: {error}")
                await self._set_status(media_file, STATUS_FAILED, count_attempt=True)
                self.stats.files_failed += 1

        logger.debug(f"Extractor {worker_id} stopped.")

    async def _infer(self) -> None:
        """Transcribe queued audio and commit the subtitle file."""
        loop = asyncio.get_running_loop()

        while True:
            job = await self.inference_queue.get()
            if job is None:
                break

            job_bytes = job.nbytes
            try:
                await self._set_status(job.media_file, STATUS_INFERENCING)
                logger.info(f"Transcribing {job.media_file.path.name}")
                if self.progress:
                    self.progress.start(job.media_file.path.name, job.media_file.duration or 0.0)

                srt_content, duration = await loop.run_in_executor(
                    self.stt.executor, self._run_inference, job, loop
                )
                self.stats.total_audio_duration_secs += duration

                if srt_content.strip():
                    await loop.run_in_executor(
                        None,
                        AtomicWriter.write_srt,
                        job.media_file.path,
                        srt_content,
                        self.subtitle_language,
                    )
                    await self._set_status(
                        job.media_file, STATUS_COMPLETED, model_used=self.stt.model_name
                    )
                    self.stats.files_processed += 1
                else:
                    # No speech is a final answer, not a failure. Recording it
                    # stops the file from being transcribed again every run.
                    logger.warning(f"No speech detected in {job.media_file.path.name}")
                    await self._set_status(job.media_file, STATUS_NO_SPEECH)
                    self.stats.files_without_speech += 1
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(f"Inference failed on {job.media_file.path.name}: {error}")
                self.stats.files_failed += 1
                await self._set_status(job.media_file, STATUS_FAILED, count_attempt=True)
            finally:
                if self.progress:
                    self.progress.finish()
                # Dropping the last reference frees the array immediately.
                # CPython uses reference counting, so no collection is needed.
                job = None
                await self.budget.release(job_bytes)

        logger.debug("Inference worker stopped.")

    def _run_inference(
        self, job: InferenceJob, loop: asyncio.AbstractEventLoop
    ) -> tuple[str, float]:
        """Transcribe one file. Runs on the model's worker thread.

        Args:
            job: The file and its decoded audio.
            loop: The event loop, used to report progress from this thread.

        Returns:
            The SRT document and the audio duration in seconds.
        """
        language = self.language
        if language is None:
            detected, confidence = self.stt.detect_language(job.audio_data)
            if detected and confidence >= _MIN_LANGUAGE_CONFIDENCE:
                language = detected
                logger.debug(f"Detected language '{detected}' ({confidence:.0%} confident).")
            elif detected:
                logger.warning(
                    f"Low confidence ({confidence:.0%}) for language '{detected}' in "
                    f"{job.media_file.path.name}. Letting the model decide per window."
                )

        report = self.progress
        on_progress: Callable[[float], None] | None = None
        if report is not None:

            def post_progress(seconds: float) -> None:
                """Hand a progress update to the event loop thread."""
                loop.call_soon_threadsafe(report.advance, seconds)

            on_progress = post_progress

        segments, duration = self.stt.transcribe(
            job.audio_data,
            translate=self.translate,
            language=language,
            on_progress=on_progress,
        )
        return self.formatter.format_segments(segments), duration

    async def _set_status(
        self,
        media_file: MediaFile,
        status: str,
        model_used: str | None = None,
        count_attempt: bool = False,
    ) -> None:
        """Record a state transition. A database error never stops the run."""
        try:
            attempts = 0
            if count_attempt:
                existing = await self.state.get_state(str(media_file.path))
                attempts = (existing.attempts if existing else 0) + 1
            await self.state.update_state(
                file_path=str(media_file.path),
                inode=media_file.inode,
                mtime=media_file.mtime,
                size=media_file.size,
                status=status,
                model_used=model_used,
                device=media_file.device,
                attempts=attempts,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(f"Could not record state '{status}' for {media_file.path.name}: {error}")


@dataclass(slots=True)
class ProgressReporter:
    """Receives progress updates on the event loop thread.

    The inference worker runs on its own thread, so it must never touch a
    terminal display directly. It posts updates here instead.
    """

    current_file: str = ""
    total_seconds: float = 0.0
    completed_seconds: float = 0.0

    def start(self, filename: str, total_seconds: float) -> None:
        """Begin reporting for a new file."""
        self.current_file = filename
        self.total_seconds = total_seconds
        self.completed_seconds = 0.0

    def advance(self, completed_seconds: float) -> None:
        """Record how far into the audio the model has reached."""
        self.completed_seconds = completed_seconds

    def finish(self) -> None:
        """Stop reporting for the current file."""
        self.current_file = ""
        self.completed_seconds = 0.0


def _estimate_bytes(duration: float | None) -> int:
    """Estimate the memory one file's decoded audio needs.

    Args:
        duration: The probed duration in seconds, or None when it is unknown.

    Returns:
        A byte count. An unknown duration is treated as one hour.
    """
    seconds = duration if duration and duration > 0 else 3600.0
    return int(seconds * SAMPLE_RATE * _BYTES_PER_SAMPLE)
