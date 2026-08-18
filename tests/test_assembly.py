"""Tests for subtitle formatting and atomic subtitle writing."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aisrt.assembly import (
    AtomicWriter,
    Cue,
    SRTFormatter,
    SubtitleStyle,
    _ends_sentence,
    format_timestamp,
    sidecar_path,
)
from tests.conftest import FakeSegment, FakeWord, SegmentFactory


class TestFormatTimestamp:
    """format_timestamp must always produce a valid SRT timecode."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.0, "00:01:01,000"),
            (3661.1234, "01:01:01,123"),
            (1.9999, "00:00:02,000"),
            (59.9999, "00:01:00,000"),
            (3599.9996, "01:00:00,000"),
            (3600.0, "01:00:00,000"),
            (360000.0, "100:00:00,000"),
        ],
    )
    def test_known_values(self, seconds: float, expected: str) -> None:
        """Each input maps to the documented timecode."""
        assert format_timestamp(seconds) == expected

    def test_negative_clamps_to_zero(self) -> None:
        """A negative time must not produce a malformed timecode."""
        assert format_timestamp(-1.5) == "00:00:00,000"
        assert format_timestamp(-0.001) == "00:00:00,000"


class TestSentenceDetection:
    """A period does not always end a sentence."""

    @pytest.mark.parametrize("token", ["end.", "Really?", "Stop!", "done…", 'said."'])
    def test_sentence_enders(self, token: str) -> None:
        """A real terminal mark ends the sentence."""
        assert _ends_sentence(token) is True

    @pytest.mark.parametrize("token", ["Mr.", "Dr.", "vs.", "U.S.", "A.", "3.5.", "word"])
    def test_not_sentence_enders(self, token: str) -> None:
        """An abbreviation, an initial, or a decimal does not."""
        assert _ends_sentence(token) is False


class TestSRTFormatterCompliance:
    """Generated cues must respect every layout and timing limit."""

    @staticmethod
    def _assert_compliant(cues: list[Cue], style: SubtitleStyle) -> None:
        """Check one cue list against the style limits."""
        previous: Cue | None = None
        for index, cue in enumerate(cues, start=1):
            lines = cue.text.split("\n")
            assert len(lines) <= style.max_lines, f"cue {index} has {len(lines)} lines"
            for line in lines:
                assert line == line.strip(), f"cue {index} line has stray whitespace: {line!r}"
                if " " in line:
                    assert len(line) <= style.max_chars_per_line, (
                        f"cue {index} line is {len(line)} characters: {line!r}"
                    )
            assert cue.start >= 0.0
            assert cue.end > cue.start, f"cue {index} has no duration"
            assert cue.duration <= style.max_duration + 1e-6, f"cue {index} runs too long"
            assert cue.cps <= style.max_cps + 1e-6, f"cue {index} reads at {cue.cps:.1f} cps"
            if previous is not None:
                gap = cue.start - previous.end
                assert gap >= style.min_gap - 1e-6, f"cue {index} overlaps its predecessor"
            previous = cue

    def test_long_passage_is_compliant(self, make_segment: SegmentFactory) -> None:
        """A long passage splits into cues that break no limit."""
        text = (
            "The quick brown fox jumps over the lazy dog and then it runs away into "
            "the deep dark forest without making any sound at all tonight. But we "
            "must go on because the night is long and the road ahead is very dark. "
            "Mr. Smith went to Washington in 1939 and it cost 3.5 million dollars."
        )
        formatter = SRTFormatter()
        cues = formatter.build_cues([make_segment(text)])
        assert len(cues) > 3
        self._assert_compliant(cues, formatter.style)

    def test_lines_respect_the_character_limit(self, make_segment: SegmentFactory) -> None:
        """Spaces between words count toward the line length."""
        text = "It is a very good day to be out here in the park with all of my friends today"
        formatter = SRTFormatter(max_chars_per_line=42, max_lines=2)
        cues = formatter.build_cues([make_segment(text, seconds_per_word=0.3)])
        for cue in cues:
            for line in cue.text.split("\n"):
                assert len(line) <= 42

    def test_line_break_avoids_a_dangling_preposition(self, make_segment: SegmentFactory) -> None:
        """A line must not end on an article, preposition, or auxiliary."""
        text = "It is a very good day to be out here in the park with all of my friends"
        formatter = SRTFormatter()
        cues = formatter.build_cues([make_segment(text, seconds_per_word=0.3)])
        for cue in cues:
            lines = cue.text.split("\n")
            for line in lines[:-1]:
                assert line.split()[-1].lower() not in {"a", "the", "to", "with", "of", "in"}

    def test_lines_are_balanced(self, make_segment: SegmentFactory) -> None:
        """Two lines of one cue stay close in length."""
        text = "The night is long and the road ahead is dark and cold and very lonely"
        formatter = SRTFormatter()
        cues = formatter.build_cues([make_segment(text, seconds_per_word=0.35)])
        for cue in cues:
            lines = cue.text.split("\n")
            if len(lines) == 2:
                assert abs(len(lines[0]) - len(lines[1])) <= 20

    def test_silence_forces_a_new_cue(self) -> None:
        """A pause of half a second or more starts a new cue."""
        words = [
            FakeWord("Hello", 0.0, 0.5),
            FakeWord("world.", 0.5, 1.0),
            FakeWord("Are", 5.0, 5.4),
            FakeWord("you", 5.4, 5.8),
            FakeWord("there?", 5.8, 6.3),
        ]
        segment = FakeSegment("Hello world. Are you there?", 0.0, 6.3, words)
        cues = SRTFormatter().build_cues([segment])
        assert len(cues) == 2
        assert cues[0].text == "Hello world."
        assert cues[1].text == "Are you there?"

    def test_segments_without_words_still_produce_cues(self) -> None:
        """A segment without word timings falls back to even spacing."""
        segment = FakeSegment("Hello there, friend.", 0.0, 2.0, None)
        srt = SRTFormatter().format_segments([segment])
        assert "Hello there, friend." in srt
        assert srt.startswith("1\n00:00:00,000 --> ")

    def test_empty_input_produces_empty_output(self) -> None:
        """Nothing said means nothing written."""
        assert SRTFormatter().format_segments([]) == ""
        assert SRTFormatter().format_segments([FakeSegment("   ", 0.0, 1.0, None)]) == ""

    def test_formatter_is_reusable(self, make_segment: SegmentFactory) -> None:
        """A shared formatter restarts numbering for every file."""
        formatter = SRTFormatter()
        first = formatter.format_segments([make_segment("One two three four five.")])
        second = formatter.format_segments([make_segment("Six seven eight nine ten.")])
        assert first.startswith("1\n")
        assert second.startswith("1\n")
        assert first != second

    def test_output_is_well_formed_srt(self, make_segment: SegmentFactory) -> None:
        """Blocks are separated by a blank line and the file ends with one newline."""
        srt = SRTFormatter().format_segments([make_segment("One two three. Four five six seven.")])
        assert srt.endswith("\n")
        assert not srt.endswith("\n\n")
        blocks = srt.rstrip("\n").split("\n\n")
        for number, block in enumerate(blocks, start=1):
            lines = block.split("\n")
            assert lines[0] == str(number)
            assert " --> " in lines[1]
            assert len(lines) >= 3


class TestSidecarPath:
    """The sidecar name must match what Plex and Jellyfin look for."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("Movie.2019.1080p.BluRay.mkv", "Movie.2019.1080p.BluRay.en.srt"),
            ("Show.S01E01.mkv", "Show.S01E01.en.srt"),
            ("clip.MP4", "clip.en.srt"),
            ("Movie.2019", "Movie.2019.en.srt"),
            ("2001.A.Space.Odyssey", "2001.A.Space.Odyssey.en.srt"),
        ],
    )
    def test_names(self, tmp_path: Path, filename: str, expected: str) -> None:
        """The extension is replaced only when it is a real video extension."""
        assert sidecar_path(tmp_path / filename, "en").name == expected

    def test_flags_are_appended(self, tmp_path: Path) -> None:
        """A flag sits after the language code."""
        path = sidecar_path(tmp_path / "Movie.mkv", "en", ("forced",))
        assert path.name == "Movie.en.forced.srt"

    @pytest.mark.parametrize("code", ["../evil", "en/us", "", "toolongcode"])
    def test_bad_language_codes_are_rejected(self, tmp_path: Path, code: str) -> None:
        """A code that is not a plain language tag cannot reach the filesystem."""
        with pytest.raises(ValueError, match="Invalid subtitle language code"):
            sidecar_path(tmp_path / "Movie.mkv", code)


class TestAtomicWriter:
    """Writing a subtitle must be all-or-nothing."""

    def test_writes_and_commits(self, tmp_path: Path) -> None:
        """The final file appears and no temporary file is left behind."""
        video = tmp_path / "movie.mkv"
        video.write_text("video")
        content = "1\n00:00:00,000 --> 00:00:01,000\nTest.\n"

        with patch("os.chown"):
            final = AtomicWriter.write_srt(video, content, language_code="en")

        assert final.name == "movie.en.srt"
        assert final.read_text(encoding="utf-8") == content
        assert not list(tmp_path.glob("*.tmp"))

    def test_dotted_filename_keeps_every_component(self, tmp_path: Path) -> None:
        """A release name full of dots keeps its title."""
        video = tmp_path / "Movie.2019.1080p.BluRay.x264.mkv"
        video.write_text("video")
        with patch("os.chown"):
            final = AtomicWriter.write_srt(video, "1\n00:00:00,000 --> 00:00:01,000\nHi.\n", "en")
        assert final.name == "Movie.2019.1080p.BluRay.x264.en.srt"

    def test_permissions_follow_the_source(self, tmp_path: Path) -> None:
        """The subtitle inherits the video's permission bits, minus execute."""
        video = tmp_path / "movie.mkv"
        video.write_text("video")
        os.chmod(video, 0o640)
        with patch("os.chown"):
            final = AtomicWriter.write_srt(video, "1\n00:00:00,000 --> 00:00:01,000\nHi.\n", "en")
        assert final.stat().st_mode & 0o777 == 0o640

    def test_a_chown_failure_does_not_lose_the_subtitle(self, tmp_path: Path) -> None:
        """An OSError from chown on a network share must not discard the work."""
        video = tmp_path / "movie.mkv"
        video.write_text("video")
        with patch("os.chown", side_effect=OSError(1, "Operation not permitted")):
            final = AtomicWriter.write_srt(video, "1\n00:00:00,000 --> 00:00:01,000\nHi.\n", "en")
        assert final.exists()

    def test_a_write_failure_cleans_up(self, tmp_path: Path) -> None:
        """A failed rename removes the temporary file and reports the failure."""
        video = tmp_path / "movie.mkv"
        video.write_text("video")
        with (
            patch("os.chown"),
            patch("os.replace", side_effect=OSError(28, "No space left on device")),
            pytest.raises(RuntimeError, match="Atomic subtitle write failed"),
        ):
            AtomicWriter.write_srt(video, "content", "en")
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrites_an_existing_subtitle(self, tmp_path: Path) -> None:
        """A rerun replaces the previous subtitle in one step."""
        video = tmp_path / "movie.mkv"
        video.write_text("video")
        (tmp_path / "movie.en.srt").write_text("stale")
        with patch("os.chown"):
            final = AtomicWriter.write_srt(video, "fresh", "en")
        assert final.read_text() == "fresh"
