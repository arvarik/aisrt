"""Tests for zero-disk audio extraction."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from aisrt.extractor import (
    MAX_DURATION_SECONDS,
    SAMPLE_RATE,
    AudioExtractor,
    _drain_pcm,
    _initial_capacity,
    resident_bytes,
)
from tests.conftest import ffmpeg_installed

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_installed(), reason="FFmpeg is not installed")


class FakeStream:
    """A StreamReader stand-in that hands out fixed chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Store the chunks this stream will hand out, in order."""
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        """Return the next chunk, or empty bytes at the end."""
        return self._chunks.pop(0) if self._chunks else b""


def as_reader(stream: FakeStream) -> asyncio.StreamReader:
    """Present the fake as the StreamReader the extractor expects."""
    return cast(asyncio.StreamReader, stream)


class TestInitialCapacity:
    """The buffer is sized from the probed duration."""

    def test_sizes_from_duration(self) -> None:
        """A known duration reserves that many samples plus a little slack."""
        capacity = _initial_capacity(3600.0)
        assert capacity >= 3600 * SAMPLE_RATE
        assert capacity <= 3610 * SAMPLE_RATE

    @pytest.mark.parametrize("duration", [None, 0.0, -5.0])
    def test_unknown_duration_starts_small(self, duration: float | None) -> None:
        """An unknown duration must not reserve gigabytes up front."""
        assert _initial_capacity(duration) == SAMPLE_RATE * 60

    def test_an_absurd_duration_is_capped(self) -> None:
        """A broken header cannot make the process allocate without bound."""
        capacity = _initial_capacity(10_000_000.0)
        assert capacity <= int((MAX_DURATION_SECONDS + 10) * SAMPLE_RATE)


class TestDrainPcm:
    """The reader converts 16-bit PCM into normalized float32."""

    @pytest.mark.asyncio
    async def test_converts_and_scales(self) -> None:
        """Every sample lands in the range -1.0 to 1.0."""
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        audio = await _drain_pcm(as_reader(FakeStream([samples.tobytes()])), duration=1.0)

        assert audio.dtype == np.float32
        assert audio.size == 5
        assert audio[0] == pytest.approx(0.0)
        assert audio[1] == pytest.approx(0.5)
        assert audio[3] == pytest.approx(0.99997, abs=1e-4)
        assert audio[4] == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_handles_a_sample_split_across_chunks(self) -> None:
        """An odd byte at a chunk boundary must not corrupt the stream."""
        samples = np.arange(-100, 100, dtype=np.int16)
        raw = samples.tobytes()
        chunks = [raw[:37], raw[37:150], raw[150:]]

        audio = await _drain_pcm(as_reader(FakeStream(chunks)), duration=1.0)

        assert audio.size == samples.size
        np.testing.assert_allclose(audio, samples.astype(np.float32) / 32768.0, atol=1e-7)

    @pytest.mark.asyncio
    async def test_grows_past_an_underestimated_duration(self) -> None:
        """A duration that is too small grows the buffer instead of truncating."""
        samples = np.arange(0, SAMPLE_RATE * 3, dtype=np.int16)
        audio = await _drain_pcm(as_reader(FakeStream([samples.tobytes()])), duration=0.01)
        assert audio.size == samples.size

    @pytest.mark.asyncio
    async def test_a_trailing_half_sample_is_discarded(self) -> None:
        """A truncated final sample is dropped, not turned into a ValueError."""
        raw = np.array([1, 2, 3], dtype=np.int16).tobytes() + b"\x01"
        audio = await _drain_pcm(as_reader(FakeStream([raw])), duration=1.0)
        assert audio.size == 3

    @pytest.mark.asyncio
    async def test_empty_stream_yields_an_empty_array(self) -> None:
        """No output means no samples, not an exception."""
        audio = await _drain_pcm(as_reader(FakeStream([])), duration=1.0)
        assert audio.size == 0


class TestResidentBytes:
    """The memory budget must count the whole allocation, not the visible slice."""

    def test_a_view_reports_its_base(self) -> None:
        """A slice keeps the entire buffer alive, so it must report the buffer."""
        buffer = np.empty(1000, dtype=np.float32)
        view = buffer[:100]

        assert view.nbytes == 400
        assert resident_bytes(view) == buffer.nbytes == 4000

    def test_a_standalone_array_reports_itself(self) -> None:
        """An array that owns its memory reports its own size."""
        array = np.empty(100, dtype=np.float32)
        assert resident_bytes(array) == array.nbytes == 400

    @pytest.mark.asyncio
    async def test_the_drained_array_is_a_view(self) -> None:
        """Copying would briefly hold both arrays and raise peak memory by half."""
        samples = np.arange(0, 1000, dtype=np.int16)
        audio = await _drain_pcm(as_reader(FakeStream([samples.tobytes()])), duration=60.0)

        assert audio.size == samples.size
        assert audio.base is not None, "the result was copied instead of sliced"
        assert resident_bytes(audio) > audio.nbytes


class TestExtractCommand:
    """The FFmpeg command line is the zero-disk contract."""

    @pytest.mark.asyncio
    async def test_command_line(self) -> None:
        """Every flag the pipeline depends on is present."""
        process = AsyncMock()
        process.returncode = 0
        process.stdout = FakeStream([np.zeros(SAMPLE_RATE * 2, dtype=np.int16).tobytes()])
        process.stderr = FakeStream([])
        process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
            await AudioExtractor.extract_audio_to_memory(Path("movie.mkv"), 2, duration=2.0)

        argv = list(spawn.call_args.args)
        kwargs = spawn.call_args.kwargs

        assert argv[0] == "ffmpeg"
        assert "-nostdin" in argv
        assert argv[argv.index("-map") + 1] == "0:a:2"
        assert argv[argv.index("-ac") + 1] == "1"
        assert argv[argv.index("-ar") + 1] == str(SAMPLE_RATE)
        assert argv[argv.index("-f") + 1] == "s16le"
        # A dash as the output target is what keeps the audio off the disk.
        assert argv[-1] == "-"
        assert "-vn" in argv
        assert "-sn" in argv
        assert "-dn" in argv
        assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
        assert kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_raises_with_the_stderr_text(self) -> None:
        """The FFmpeg error message reaches the caller."""
        process = AsyncMock()
        process.returncode = 1
        process.stdout = FakeStream([np.zeros(SAMPLE_RATE * 2, dtype=np.int16).tobytes()])
        process.stderr = FakeStream([b"Invalid data found when processing input"])
        process.wait = AsyncMock(return_value=1)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            pytest.raises(RuntimeError, match="Invalid data found"),
        ):
            await AudioExtractor.extract_audio_to_memory(Path("bad.mkv"), 0, duration=2.0)

    @pytest.mark.asyncio
    async def test_too_little_audio_raises(self) -> None:
        """Under a second of audio means the track is unreadable."""
        process = AsyncMock()
        process.returncode = 0
        process.stdout = FakeStream([np.zeros(100, dtype=np.int16).tobytes()])
        process.stderr = FakeStream([])
        process.wait = AsyncMock(return_value=0)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            pytest.raises(RuntimeError, match="only 100 samples"),
        ):
            await AudioExtractor.extract_audio_to_memory(Path("silent.mkv"), 0, duration=2.0)

    @pytest.mark.asyncio
    async def test_a_timeout_kills_the_process_and_raises(self) -> None:
        """A hung FFmpeg is reaped and reported, never left running."""
        process = AsyncMock()
        process.returncode = None
        process.pid = 4321

        async def never_finishes(_size: int) -> bytes:
            await asyncio.sleep(60)
            return b""

        process.stdout = AsyncMock()
        process.stdout.read = never_finishes
        process.stderr = FakeStream([])

        reaped: list[object] = []

        async def fake_reap(proc: object, grace: float = 5.0) -> None:
            reaped.append(proc)

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
            patch("aisrt.extractor.reap_process", new=fake_reap),
            pytest.raises(RuntimeError, match="timed out after 0s"),
        ):
            await AudioExtractor.extract_audio_to_memory(
                Path("hung.mkv"), 0, timeout=0.05, duration=2.0
            )

        assert reaped, "the child process was never reaped"


@needs_ffmpeg
@pytest.mark.integration
class TestExtractIntegration:
    """These tests run the real FFmpeg binary."""

    @staticmethod
    def _make_audio_file(path: Path, duration: float = 2.0) -> None:
        """Build a small file holding a sine tone."""
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={duration}",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    @pytest.mark.asyncio
    async def test_extracts_real_audio(self, tmp_path: Path) -> None:
        """The decoded array has the expected rate, length, and range."""
        media = tmp_path / "tone.m4a"
        self._make_audio_file(media, duration=2.0)

        audio = await AudioExtractor.extract_audio_to_memory(media, 0, duration=2.0)

        assert audio.dtype == np.float32
        assert audio.size == pytest.approx(2 * SAMPLE_RATE, rel=0.1)
        assert float(np.abs(audio).max()) <= 1.0
        # A 440 Hz tone is not silence.
        assert float(np.abs(audio).mean()) > 0.01

    @pytest.mark.asyncio
    async def test_an_unknown_duration_still_extracts(self, tmp_path: Path) -> None:
        """The buffer grows correctly when the duration was not probed."""
        media = tmp_path / "tone.m4a"
        self._make_audio_file(media, duration=3.0)

        audio = await AudioExtractor.extract_audio_to_memory(media, 0, duration=None)

        assert audio.size == pytest.approx(3 * SAMPLE_RATE, rel=0.1)

    @pytest.mark.asyncio
    async def test_a_missing_track_fails_cleanly(self, tmp_path: Path) -> None:
        """Asking for a track that does not exist raises, and does not hang."""
        media = tmp_path / "tone.m4a"
        self._make_audio_file(media)

        with pytest.raises(RuntimeError):
            await AudioExtractor.extract_audio_to_memory(media, 7, duration=2.0)
