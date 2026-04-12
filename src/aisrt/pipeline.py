"""Asynchronous TaskGroup pipeline for Producer-Consumer workflow."""

import asyncio
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from aisrt.discovery import DiscoveryEngine, MediaFile
from aisrt.extractor import AudioExtractor


@dataclass
class PipelineStats:
    """Statistics for a single pipeline run."""
    files_scanned: int = 0
    files_skipped: int = 0
    files_processed: int = 0
    files_failed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    total_audio_duration_secs: float = 0.0


@dataclass
class InferenceJob:
    """A media file packaged with its loaded audio data."""

    media_file: MediaFile
    audio_data: np.ndarray


class Pipeline:
    """Manages the bounded queues and concurrent workers for the STT pipeline."""

    def __init__(
        self, discovery_engine: DiscoveryEngine, cpu_cores: int, translate: bool = False
    ) -> None:
        """Initialize the pipeline.

        Args:
            discovery_engine: The NAS-safe file crawler.
            cpu_cores: Used to set the maxsize of the extraction queue.
            translate: If True, uses Whisper to translate foreign audio to English.
        """
        self.discovery = discovery_engine
        self.target_languages = discovery_engine.config.target_languages
        self.translate = translate

        # Bounded queues prevent Out-Of-Memory errors and NAS I/O thrashing
        # GPU inference is the true bottleneck, so extraction only needs a tiny buffer
        max_extractors = min(cpu_cores, 3)
        self.extraction_queue: asyncio.Queue[MediaFile | None] = asyncio.Queue(
            maxsize=max_extractors
        )
        self.inference_queue: asyncio.Queue[InferenceJob | None] = asyncio.Queue(maxsize=2)

        self.cpu_cores = cpu_cores
        self.stats = PipelineStats()

    async def run(self) -> PipelineStats:
        """Start the entire pipeline and wait for completion."""
        logger.info("Initializing async TaskGroup pipeline...")
        self.stats.start_time = time.time()
        
        async with asyncio.TaskGroup() as tg:
            # We use a supervisor task to manage the graceful teardown of workers
            tg.create_task(self._orchestrator(tg))

        self.stats.end_time = time.time()
        return self.stats

    async def _orchestrator(self, tg: asyncio.TaskGroup) -> None:
        """Supervises the tasks and issues sentinels for graceful shutdown."""
        producer = tg.create_task(self._producer())

        extractors = [
            tg.create_task(self._extractor_worker(i)) for i in range(self.extraction_queue.maxsize)
        ]

        inference = tg.create_task(self._inference_worker())

        # Wait for the producer to finish finding files
        await producer

        # Now wait for all extractors to finish their current files and hit the None sentinels
        for ext in extractors:
            await ext

        # Finally, tell the Inference GPU worker to shut down after finishing its queue
        await self.inference_queue.put(None)
        await inference

    async def _producer(self) -> None:
        """Crawls the filesystem and pushes files to the extraction queue."""
        logger.debug("Producer (Scanner) started.")
        async for media_file, action in self.discovery.scan():
            self.stats.files_scanned += 1
            if action == "PROCESS":
                await self.extraction_queue.put(media_file)
            else:
                self.stats.files_skipped += 1

        # Broadcast termination sentinels to all extractors
        logger.debug("Producer finished. Sending shutdown sentinels to extractors.")
        for _ in range(self.extraction_queue.maxsize):
            await self.extraction_queue.put(None)

    async def _extractor_worker(self, worker_id: int) -> None:
        """Pops files, extracts audio to memory, and pushes to inference."""
        logger.debug(f"Extractor {worker_id} started.")
        while True:
            media_file = await self.extraction_queue.get()
            if media_file is None:
                self.extraction_queue.task_done()
                break

            try:
                await self.discovery.state_tracker.update_state(
                    file_path=str(media_file.path),
                    inode=media_file.inode,
                    mtime=media_file.mtime,
                    size=media_file.size,
                    status="EXTRACTING",
                )

                track_idx = await AudioExtractor.get_audio_track_index(
                    media_file.path, self.target_languages
                )
                audio_data = await AudioExtractor.extract_audio_to_memory(
                    media_file.path, track_idx
                )

                # Push to GPU queue. This will block (backpressure) if the GPU is busy.
                await self.inference_queue.put(InferenceJob(media_file, audio_data))

            except Exception as e:
                logger.error(f"Extractor {worker_id} failed on {media_file.path.name}: {e}")
                # We do not crash the pipeline on a bad MKV.
            finally:
                self.extraction_queue.task_done()

        logger.debug(f"Extractor {worker_id} cleanly shut down.")

    async def _inference_worker(self) -> None:
        """Pops NumPy arrays and performs AI inference."""
        from aisrt.assembly import AtomicWriter, SRTFormatter
        from aisrt.stt import STTWorker

        logger.debug("Inference worker (GPU/Singleton) started.")
        stt_worker = STTWorker()
        formatter = SRTFormatter()
        loop = asyncio.get_running_loop()

        while True:
            job = await self.inference_queue.get()
            if job is None:
                self.inference_queue.task_done()
                break

            try:
                await self.discovery.state_tracker.update_state(
                    file_path=str(job.media_file.path),
                    inode=job.media_file.inode,
                    mtime=job.media_file.mtime,
                    size=job.media_file.size,
                    status="INFERENCING",
                )
                logger.info(f"GPU processing: {job.media_file.path.name}...")

                # Execute inference in the dedicated STT thread pool
                def _run_inference(current_job: InferenceJob) -> str | None:
                    if not stt_worker.model:
                        raise RuntimeError("Whisper model is not initialized.")

                    transcribe_kwargs = {
                        "task": "translate" if self.translate else "transcribe",
                        "beam_size": 5,
                        "vad_filter": True,
                        "vad_parameters": {"min_silence_duration_ms": 500},
                        "condition_on_previous_text": False,
                        "compression_ratio_threshold": 2.4,
                        "no_speech_threshold": 0.6,
                        "word_timestamps": True,
                        "initial_prompt": "A well-punctuated English subtitle.",
                    }

                    if stt_worker.model.__class__.__name__ == "BatchedInferencePipeline":
                        transcribe_kwargs["batch_size"] = 16

                    segments, info = stt_worker.model.transcribe(
                        current_job.audio_data, **transcribe_kwargs
                    )
                    self.stats.total_audio_duration_secs += info.duration

                    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
                    from aisrt.cli import console

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            f"Transcribing {current_job.media_file.path.name}...", 
                            total=info.duration
                        )

                        def _segment_generator():
                            for segment in segments:
                                progress.update(task, completed=segment.end)
                                yield segment
                        
                        return formatter.format_segments(_segment_generator())

                srt_content = await loop.run_in_executor(stt_worker.executor, _run_inference, job)

                if srt_content and srt_content.strip():
                    await loop.run_in_executor(
                        None,
                        AtomicWriter.write_srt,
                        job.media_file.path,
                        srt_content,
                        self.target_languages[0] if self.target_languages else "en",
                    )
                    # Update SQLite state to COMPLETED
                    if stt_worker.model is None:
                        model_str = "unknown"
                    elif stt_worker.model.__class__.__name__ == "BatchedInferencePipeline":
                        base_model = getattr(stt_worker.model, "model", stt_worker.model)
                        model_str = getattr(base_model, "model_size_or_path", "unknown")
                    else:
                        model_str = getattr(stt_worker.model, "model_size_or_path", "unknown")
                    await self.discovery.state_tracker.update_state(
                        file_path=str(job.media_file.path),
                        inode=job.media_file.inode,
                        mtime=job.media_file.mtime,
                        size=job.media_file.size,
                        status="COMPLETED",
                        model_used=model_str,
                    )
                    self.stats.files_processed += 1
                else:
                    logger.warning(f"No speech detected in {job.media_file.path.name}")
                    self.stats.files_failed += 1

            except Exception as e:
                logger.error(f"Inference failed on {job.media_file.path.name}: {e}")
                self.stats.files_failed += 1
            finally:
                self.inference_queue.task_done()
                
                # Clean up memory explicitly
                del job.audio_data
                import gc
                gc.collect()

        logger.debug("Inference worker cleanly shut down.")
