"""Tests for the STT atomic writer and SRT formatter."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from aisrt.assembly import AtomicWriter, SRTFormatter, _format_timestamp


@dataclass
class MockWord:
    """Mock for faster-whisper Word object."""

    word: str
    start: float
    end: float


@dataclass
class MockSegment:
    """Mock for faster-whisper Segment object."""

    text: str
    start: float
    end: float
    words: list[MockWord] | None = None


def test_format_timestamp() -> None:
    """Test SRT timestamp formatting edge cases."""
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(1.5) == "00:00:01,500"
    assert _format_timestamp(61.0) == "00:01:01,000"
    assert _format_timestamp(3661.1234) == "01:01:01,123"

    # Check rounding
    assert _format_timestamp(1.9999) == "00:00:02,000"


def test_srt_formatter_basic() -> None:
    """Test the basic fallback formatter without word timestamps."""
    formatter = SRTFormatter()

    segments = [
        MockSegment("Hello world.", 1.0, 2.5),
        MockSegment("This is a test.", 3.0, 4.5),
    ]

    srt_content = formatter.format_segments(segments)

    assert "1\n00:00:01,000 --> 00:00:02,500\nHello world.\n" in srt_content
    assert "2\n00:00:03,000 --> 00:00:04,500\nThis is a test.\n" in srt_content


def test_srt_formatter_word_chunking() -> None:
    """Test the advanced broadcast chunker using word timestamps."""
    formatter = SRTFormatter(max_chars_per_line=15, max_lines=2)

    # 26 characters total. Should split after 'sentence '
    # Line 1: 'This is a long ' (15 chars)
    # Line 2: 'sentence indeed.' (16 chars, terminal)
    words = [
        MockWord("This ", 0.0, 0.5),
        MockWord("is ", 0.5, 1.0),
        MockWord("a ", 1.0, 1.5),
        MockWord("long ", 1.5, 2.0),
        MockWord(" sentence ", 2.0, 2.5),  # Testing leading space wrap
        MockWord("indeed.", 2.5, 3.0),
    ]

    segments = [MockSegment("This is a long sentence indeed.", 0.0, 3.0, words=words)]

    srt_content = formatter.format_segments(segments)

    expected_block = "1\n00:00:00,000 --> 00:00:03,000\nThis is a long \nsentence indeed.\n"
    assert expected_block in srt_content


def test_srt_formatter_temporal_gap() -> None:
    """Test that a temporal gap > 1.5s forces a subtitle flush."""
    formatter = SRTFormatter(max_chars_per_line=40, max_lines=2)

    words = [
        MockWord("Hello ", 0.0, 1.0),
        MockWord("world, ", 1.0, 2.0),
        # 3.0 second gap here
        MockWord("are ", 5.0, 5.5),
        MockWord("you ", 5.5, 6.0),
        MockWord("there?", 6.0, 6.5),
    ]

    segments = [MockSegment("Hello world, are you there?", 0.0, 6.5, words=words)]

    srt_content = formatter.format_segments(segments)

    # Should be split into two blocks due to the gap
    assert "1\n00:00:00,000 --> 00:00:02,000\nHello world," in srt_content
    assert "2\n00:00:05,000 --> 00:00:06,500\nare you there?" in srt_content


def test_atomic_writer_success(tmp_path: Path) -> None:
    """Test that the atomic writer creates temp files and replaces successfully."""
    source_video = tmp_path / "movie.mkv"
    source_video.write_text("dummy video")

    content = "1\n00:00:00,000 --> 00:00:01,000\nTest subtitle.\n"

    # We mock os.chown because it will fail in most CI/test environments unless run as root.
    with patch("os.chown"):
        final_path = AtomicWriter.write_srt(source_video, content, language_code="en")

        assert final_path.exists()
        assert final_path.name == "movie.en.srt"
        assert final_path.read_text(encoding="utf-8") == content

        # Temp file should be gone. We check that no .tmp files remain in the directory.
        temp_files = [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
        assert len(temp_files) == 0, f"Temporary files found: {temp_files}"
