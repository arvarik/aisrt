"""Tests for the concurrent Pipeline orchestrator."""

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from srtgen.config import FilterConfig
from srtgen.discovery import DiscoveryEngine, MediaFile
from srtgen.pipeline import Pipeline


@pytest.fixture
def mock_discovery() -> DiscoveryEngine:
    """Provide a DiscoveryEngine that yields mock files."""
    engine = AsyncMock(spec=DiscoveryEngine)
    engine.config = FilterConfig(target_languages=["eng"])
    engine.state_tracker = AsyncMock()
    engine.state_tracker.update_state = AsyncMock()

    async def mock_scan() -> AsyncGenerator[tuple[MediaFile, str], None]:
        yield MediaFile(Path("movie1.mkv"), 100, 1.0, 1), "PROCESS"
        yield MediaFile(Path("movie2.mkv"), 100, 1.0, 2), "PROCESS"
        yield MediaFile(Path("movie3.mkv"), 100, 1.0, 3), "SKIP"

    engine.scan = mock_scan
    return engine


@pytest.mark.asyncio
async def test_pipeline_execution(mock_discovery: DiscoveryEngine) -> None:
    """Test that the orchestrator routes tasks through extractors and inference correctly."""
    pipeline = Pipeline(mock_discovery, cpu_cores=2)

    with patch("srtgen.pipeline.AudioExtractor") as mock_ext:
        mock_ext.get_audio_track_index = AsyncMock(return_value=0)
        mock_ext.extract_audio_to_memory = AsyncMock(return_value=np.zeros(10))

        completed_jobs = []

        async def mock_inference() -> None:
            """Mock the GPU inference worker to simply collect the jobs."""
            while True:
                job = await pipeline.inference_queue.get()
                if job is None:
                    pipeline.inference_queue.task_done()
                    break
                completed_jobs.append(job)
                pipeline.inference_queue.task_done()

        # Override inference worker for test
        pipeline._inference_worker = mock_inference  # type: ignore[method-assign]

        await pipeline.run()

        assert len(completed_jobs) == 2

        # Test order could theoretically vary slightly depending on task scheduling,
        # but with small counts they are typically sequential.
        names = {job.media_file.path.name for job in completed_jobs}
        assert "movie1.mkv" in names
        assert "movie2.mkv" in names
        assert "movie3.mkv" not in names
