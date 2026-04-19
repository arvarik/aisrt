"""Broadcast-quality SubRip (SRT) formatting and Atomic File I/O."""

import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger


def _format_timestamp(seconds: float) -> str:
    """Format a timestamp (in float seconds) to SRT standard: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis == 1000:
        secs += 1
        millis = 0
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SRTFormatter:
    """Chunks Whisper words into broadcast-standard SRT format."""

    def __init__(self, max_chars_per_line: int = 42, max_lines: int = 2) -> None:
        """Initialize the SRT chunker.

        Args:
            max_chars_per_line: Maximum characters before wrapping a line.
            max_lines: Maximum lines per subtitle block.
        """
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.terminal_punctuation = {".", "?", "!", "。", "？", "！"}

    def format_segments(self, segments: Any) -> str:
        """Iterate over faster-whisper Segment/Word objects and yield SRT blocks.

        Requires word_timestamps=True in the Whisper model transcribe() call.

        Args:
            segments: A generator of faster-whisper Segment objects.

        Returns:
            The complete SRT file content as a string.
        """
        self._srt_blocks: list[str] = []
        self._block_idx = 1

        for segment in segments:
            if not getattr(segment, "words", None):
                self._format_raw_segment(segment)
            else:
                self._format_word_segment(segment)

        return "\n".join(self._srt_blocks)

    def _format_raw_segment(self, segment: Any) -> None:
        """Fallback formatter for segments without word timestamps."""
        text = segment.text.strip()
        if text:
            start = _format_timestamp(segment.start)
            end = _format_timestamp(segment.end)
            self._srt_blocks.append(f"{self._block_idx}\n{start} --> {end}\n{text}\n")
            self._block_idx += 1

    def _format_word_segment(self, segment: Any) -> None:
        """Advanced formatter that chunks based on character count and punctuation."""
        current_words: list[str] = []
        current_start: float | None = None
        current_end: float = 0.0
        char_count = 0
        line_count = 1

        for word_obj in segment.words:
            word = word_obj.word.strip()
            if not word:
                continue

            # Temporal gap check: flush if silence > 1.5s
            if current_end > 0.0 and (word_obj.start - current_end) > 1.5:
                if current_words and current_start is not None:
                    self._flush_words(current_words, current_start, current_end)
                    current_words = []
                    current_start = None
                    char_count = 0
                    line_count = 1

            if current_start is None:
                current_start = word_obj.start

            current_words.append(word_obj.word)
            current_end = word_obj.end
            char_count += len(word)

            is_terminal = any(word.endswith(p) for p in self.terminal_punctuation)

            # If appending this word exceeds the line length, wrap BEFORE adding it
            if char_count > self.max_chars_per_line and line_count < self.max_lines:
                # Insert newline before the current word
                current_words.pop()  # Remove the word we just added
                current_words.append("\n")
                current_words.append(word_obj.word.lstrip())
                char_count = len(word)
                line_count += 1

            is_too_long = char_count >= self.max_chars_per_line

            if is_terminal or (is_too_long and line_count >= self.max_lines):
                self._flush_words(current_words, current_start, current_end)
                current_words = []
                current_start = None
                char_count = 0
                line_count = 1

        if current_words and current_start is not None:
            self._flush_words(current_words, current_start, current_end)

    def _flush_words(self, words: list[str], start_time: float, end_time: float) -> None:
        """Write the aggregated words to the block list."""
        text = "".join(words).strip()
        if text:
            start_str = _format_timestamp(start_time)
            end_str = _format_timestamp(end_time)
            self._srt_blocks.append(f"{self._block_idx}\n{start_str} --> {end_str}\n{text}\n")
            self._block_idx += 1


class AtomicWriter:
    """Handles cross-device POSIX atomic file writing and metadata inheritance."""

    @staticmethod
    def write_srt(source_video: Path, srt_content: str, language_code: str = "en") -> Path:
        """Write the SRT securely, inheriting the permissions of the source video.

        Args:
            source_video: The original MKV/MP4 file.
            srt_content: The fully formatted SRT text block.
            language_code: The locale suffix for the subtitle (e.g., 'en', 'eng').

        Returns:
            The Path to the finalized, atomically committed SRT file.
        """
        final_srt_path = source_video.with_suffix(f".{language_code}.srt")
        temp_srt_path = source_video.with_name(f".{source_video.stem}.{uuid.uuid4().hex}.srt.tmp")

        logger.debug(f"Assembling atomic SRT chunks in {temp_srt_path}")

        try:
            # 1. Write to hidden temp file in the same directory
            temp_srt_path.write_text(srt_content, encoding="utf-8")

            # 2. Inherit metadata from the source video
            stat = source_video.stat()

            try:
                os.chown(temp_srt_path, stat.st_uid, stat.st_gid)
            except PermissionError:
                # Running as non-root over SMB/NFS might restrict chown
                logger.debug(
                    f"Insufficient permissions to chown {temp_srt_path} to "
                    f"UID:{stat.st_uid}/GID:{stat.st_gid}. Proceeding anyway."
                )

            try:
                os.chmod(temp_srt_path, stat.st_mode)
            except PermissionError:
                logger.debug(f"Insufficient permissions to chmod {temp_srt_path}")

            # 3. Cross-device safe Atomic Rename
            # os.replace is atomic on POSIX if both files are on the same filesystem.
            # We write the temp file in the same folder to guarantee this and prevent EXDEV errors.
            os.replace(temp_srt_path, final_srt_path)
            logger.info(f"Successfully generated and committed {final_srt_path.name}")
            return final_srt_path

        except Exception as e:
            # Clean up the temp file if the atomic commit fails
            if temp_srt_path.exists():
                try:
                    temp_srt_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Atomic subtitle write failed for {source_video.name}: {e}") from e
