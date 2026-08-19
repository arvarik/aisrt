"""Broadcast-quality SubRip (SRT) assembly and atomic file I/O.

The formatter turns word-level Whisper timestamps into cues that follow the
Netflix Timed Text Style Guide: at most 42 characters per line, at most two
lines, at most 20 characters per second, a minimum cue duration of 5/6 s, a
maximum of 7 s, and a minimum gap of two frames between neighbouring cues.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from loguru import logger

# Video containers whose extension we replace when we build a sidecar name.
VIDEO_EXTENSIONS = frozenset(
    {
        ".mkv",
        ".mp4",
        ".m4v",
        ".avi",
        ".mov",
        ".webm",
        ".ts",
        ".m2ts",
        ".mts",
        ".vob",
        ".mpg",
        ".mpeg",
        ".wmv",
        ".flv",
        ".ogv",
        ".divx",
    }
)

_TERMINAL_PUNCTUATION = frozenset(".?!。？！…")
_SOFT_PUNCTUATION = frozenset(",;:、，；：—-")
_CLOSING_MARKS = "\"')]}»”’"

# Tokens that must not end a subtitle line (Netflix / BBC line-break rules).
_NO_LINE_END = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "for",
        "so",
        "yet",
        "because",
        "that",
        "which",
        "while",
        "when",
        "if",
        "as",
        "at",
        "by",
        "in",
        "of",
        "on",
        "to",
        "up",
        "via",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "after",
        "before",
        "between",
        "during",
        "through",
        "toward",
        "towards",
        "upon",
        "within",
        "without",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "this",
        "these",
        "those",
        "not",
        "no",
    ]
)

# Words after which a period does not end a sentence.
_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "sr",
        "jr",
        "no",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "dept",
        "est",
        "fig",
        "approx",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    ]
)

_MAX_FILENAME_BYTES = 255
"""The name limit on ext4, APFS, XFS, NTFS, and most network shares."""

_INITIAL_RE = re.compile(r"^[A-Z]\.$")
_DOTTED_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")
_DECIMAL_RE = re.compile(r"^\d+(?:\.\d+)*\.$")
_LANGUAGE_CODE_RE = re.compile(r"\A[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*\Z")


@dataclass(frozen=True, slots=True)
class SubtitleStyle:
    """Broadcast layout and timing limits applied to every generated cue."""

    max_chars_per_line: int = 42
    max_lines: int = 2
    max_cps: float = 20.0
    min_duration: float = 5.0 / 6.0
    max_duration: float = 7.0
    min_gap: float = 0.084
    gap_snap_max: float = 0.5
    silence_split: float = 0.5
    lead_out: float = 0.5
    lead_in: float = 0.12
    """How far a cue may start before its first word. Netflix allows one or two
    frames. More than that shows the text before the actor speaks."""

    @property
    def max_block_chars(self) -> int:
        """Maximum number of visible characters in one cue."""
        return self.max_chars_per_line * self.max_lines


@dataclass(slots=True)
class Word:
    """One timed token taken from Whisper's word-level output."""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class Cue:
    """One finished subtitle event."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Length of the cue in seconds."""
        return self.end - self.start

    @property
    def char_count(self) -> int:
        """Number of visible characters, excluding line breaks."""
        return len(self.text.replace("\n", " "))

    @property
    def cps(self) -> float:
        """Reading speed in characters per second."""
        return self.char_count / self.duration if self.duration > 0 else float("inf")


def format_timestamp(seconds: float) -> str:
    """Format a time in seconds as an SRT timestamp (``HH:MM:SS,mmm``).

    Args:
        seconds: The time offset. Negative values clamp to zero.

    Returns:
        The timestamp string. Hours above 99 are not truncated.
    """
    total_ms = round(max(0.0, seconds) * 1000.0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _strip_closing(token: str) -> str:
    """Remove trailing quotes and brackets so punctuation tests see the mark."""
    return token.rstrip(_CLOSING_MARKS)


def _ends_sentence(token: str) -> bool:
    """Report whether the token ends a sentence rather than an abbreviation."""
    core = _strip_closing(token)
    if not core or core[-1] not in _TERMINAL_PUNCTUATION:
        return False
    if core[-1] != ".":
        return True
    if _INITIAL_RE.match(core) or _DOTTED_RE.match(core) or _DECIMAL_RE.match(core):
        return False
    return core[:-1].lower() not in _ABBREVIATIONS


def _ends_clause(token: str) -> bool:
    """Report whether the token ends a clause (comma, colon, semicolon, dash)."""
    core = _strip_closing(token)
    return bool(core) and core[-1] in _SOFT_PUNCTUATION


def _is_weak_line_end(token: str) -> bool:
    """Report whether ending a line on this token breaks a grammatical unit."""
    return _strip_closing(token).lower().strip(".,;:!?") in _NO_LINE_END


class SRTFormatter:
    """Converts Whisper segments into standards-compliant SRT text."""

    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        style: SubtitleStyle | None = None,
    ) -> None:
        """Initialize the formatter.

        Args:
            max_chars_per_line: Maximum characters on one subtitle line.
            max_lines: Maximum lines in one cue.
            style: A complete style object. It overrides the two arguments above.
        """
        self.style = style or SubtitleStyle(
            max_chars_per_line=max_chars_per_line, max_lines=max_lines
        )

    @property
    def max_chars_per_line(self) -> int:
        """Maximum characters on one subtitle line."""
        return self.style.max_chars_per_line

    @property
    def max_lines(self) -> int:
        """Maximum lines in one cue."""
        return self.style.max_lines

    def format_segments(self, segments: Iterable[Any]) -> str:
        """Convert Whisper segments into the full text of an SRT file.

        The method holds no state between calls, so one formatter instance is
        safe to reuse for every file in a run.

        Args:
            segments: An iterable of faster-whisper ``Segment`` objects. Word
                timestamps are used when present, otherwise the segment text is
                spread evenly across the segment duration.

        Returns:
            The complete SRT document. An empty string when nothing was said.
        """
        return self.to_srt(self.build_cues(segments))

    def build_cues(self, segments: Iterable[Any]) -> list[Cue]:
        """Build the list of finished cues for the given segments."""
        words = _sanitize(_flatten_segments(segments))
        if not words:
            return []
        cues: list[Cue] = []
        for group in self._split_sentences(words):
            cues.extend(self._split_group(group))
        cues = self._merge_short(cues)
        self._retime(cues)
        self._normalize_gaps(cues)
        # A cue can still be unreadably short when its neighbours leave no room.
        # Merging it into a neighbour is the only remaining repair.
        for _ in range(3):
            if not any(c.duration < self.style.min_duration - 1e-6 for c in cues):
                break
            merged = self._merge_short(cues, force=True)
            if len(merged) == len(cues):
                break
            cues = merged
            self._retime(cues)
            self._normalize_gaps(cues)
        return [cue for cue in cues if cue.text.strip() and cue.duration > 0]

    def to_srt(self, cues: Sequence[Cue]) -> str:
        """Serialize cues to SRT text with LF endings and a trailing newline."""
        if not cues:
            return ""
        blocks = [
            f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
            for index, cue in enumerate(cues, start=1)
        ]
        return "\n\n".join(blocks) + "\n"

    # --- Pass 2: sentence and silence segmentation ---

    def _split_sentences(self, words: list[Word]) -> Iterator[list[Word]]:
        """Cut the word stream after sentences and at long silences."""
        group: list[Word] = []
        for word in words:
            if group:
                silence = word.start - group[-1].end
                if silence >= self.style.silence_split or _ends_sentence(group[-1].text):
                    yield group
                    group = []
            group.append(word)
        if group:
            yield group

    # --- Pass 3: split a sentence into cue-sized pieces ---

    def _split_group(self, words: list[Word]) -> list[Cue]:
        """Split one sentence into pieces that satisfy every layout limit."""
        pieces = [words]
        result: list[Cue] = []
        while pieces:
            piece = pieces.pop(0)
            if self._fits(piece):
                result.append(self._make_cue(piece))
                continue
            index = self._best_split(piece)
            if index is None:
                result.append(self._make_cue(piece))
                continue
            pieces.insert(0, piece[index:])
            pieces.insert(0, piece[:index])
        return result

    def _fits(self, words: list[Word]) -> bool:
        """Report whether the words fit one cue without breaking any limit."""
        if len(words) <= 1:
            return True
        chars = _char_count(words)
        duration = words[-1].end - words[0].start
        return (
            chars <= self.style.max_block_chars
            and duration <= self.style.max_duration
            and _min_lines([w.text for w in words], self.style.max_chars_per_line)
            <= self.style.max_lines
        )

    def _best_split(self, words: list[Word]) -> int | None:
        """Choose the index that splits the words at the most natural point.

        Returns:
            The index of the first word of the second piece, or None when no
            split keeps both pieces usable.
        """
        count = len(words)
        if count < 2:
            return None
        chars = _char_count(words)
        duration = max(words[-1].end - words[0].start, 1e-6)
        pieces = max(
            2,
            math.ceil(chars / self.style.max_block_chars),
            math.ceil(duration / self.style.max_duration),
            math.ceil(chars / (self.style.max_cps * duration)),
        )
        ideal = count / pieces
        best_index: int | None = None
        best_score = float("-inf")
        for index in range(1, count):
            if _char_count(words[:index]) > self.style.max_block_chars:
                # The left piece is already over budget. A single word longer
                # than one cue makes this true at index 1, so keep the first
                # candidate rather than abandoning the split: one oversized word
                # must not force the whole sentence into a single cue.
                if best_index is None:
                    best_index = index
                break
            score = _split_score(words, index)
            score -= abs(index - ideal) * 3.0
            if _char_count(words[index:]) > self.style.max_block_chars:
                score -= 5.0
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _make_cue(self, words: list[Word]) -> Cue:
        """Wrap the words into lines and produce a cue."""
        text = self._wrap([w.text for w in words])
        return Cue(start=words[0].start, end=words[-1].end, text=text)

    # --- Pass 4: merge cues that are too short to read ---

    def _merge_short(self, cues: list[Cue], force: bool = False) -> list[Cue]:
        """Merge neighbouring cues while the result stays inside every limit.

        Args:
            cues: The cues to sweep, in playback order.
            force: If True, also merge across a sentence boundary. Use this only
                after timing has run and a cue is still below the minimum
                duration, because a readable cue matters more than a clean cut.
        """
        if len(cues) < 2:
            return cues
        merged: list[Cue] = [cues[0]]
        for cue in cues[1:]:
            previous = merged[-1]
            combined_chars = previous.char_count + 1 + cue.char_count
            combined_duration = cue.end - previous.start
            gap = cue.start - previous.end
            too_short = (
                previous.duration < self.style.min_duration
                or cue.duration < self.style.min_duration
            )
            ends_sentence = _ends_sentence(_tokens(previous.text)[-1])
            if (
                too_short
                and (force or not ends_sentence)
                and combined_chars <= self.style.max_block_chars
                and combined_duration <= self.style.max_duration
                and gap < self.style.silence_split
                and combined_chars / max(combined_duration, 1e-6) <= self.style.max_cps
            ):
                tokens = _tokens(previous.text) + _tokens(cue.text)
                if _min_lines(tokens, self.style.max_chars_per_line) <= self.style.max_lines:
                    merged[-1] = Cue(previous.start, cue.end, self._wrap(tokens))
                    continue
            merged.append(cue)
        return merged

    # --- Pass 5: line wrapping ---

    def _wrap(self, tokens: Sequence[str]) -> str:
        """Lay the tokens out over at most ``max_lines`` balanced lines."""
        clean = [t for t in (token.strip() for token in tokens) if t]
        if not clean:
            return ""
        limit = self.style.max_chars_per_line
        lines_needed = min(_min_lines(clean, limit), self.style.max_lines)
        if lines_needed <= 1:
            return " ".join(clean)
        layout = _best_layout(clean, lines_needed, limit)
        return "\n".join(layout)

    # --- Pass 6: per-cue timing ---

    def _retime(self, cues: list[Cue]) -> None:
        """Give every cue a readable duration inside the free time available."""
        style = self.style
        for index, cue in enumerate(cues):
            floor = cues[index - 1].end + style.min_gap if index else 0.0
            ceiling = cues[index + 1].start - style.min_gap if index + 1 < len(cues) else None
            spoken_start = cue.start

            cue.start = max(cue.start, floor, 0.0)
            cue.end = max(cue.end, cue.start)

            wanted = max(style.min_duration, cue.char_count / style.max_cps)
            wanted = min(wanted, style.max_duration)

            room_after = (
                style.lead_out if ceiling is None else min(style.lead_out, ceiling - cue.end)
            )
            cue.end += max(0.0, room_after)

            if cue.duration < wanted:
                target = cue.start + wanted
                cue.end = target if ceiling is None else min(target, max(ceiling, cue.end))
            if cue.duration < wanted:
                # Pull the in-time earlier only within the lead-in allowance.
                # Beyond that the subtitle would appear before it is spoken.
                earliest = max(floor, spoken_start - style.lead_in, 0.0)
                cue.start = max(earliest, cue.end - wanted)
            if cue.duration > style.max_duration:
                cue.end = cue.start + style.max_duration
            if cue.end <= cue.start:
                cue.end = cue.start + style.min_duration

    # --- Pass 7: gap normalization ---

    def _normalize_gaps(self, cues: list[Cue]) -> None:
        """Force every neighbouring pair apart by exactly one minimum gap."""
        style = self.style
        for current, following in pairwise(cues):
            gap = following.start - current.end
            if gap >= style.gap_snap_max:
                continue
            # Netflix chaining: any gap under half a second closes to exactly
            # the minimum gap, which removes the flicker between neighbours.
            target = min(following.start - style.min_gap, current.start + style.max_duration)
            current.end = max(target, current.start + 0.001)


def _flatten_segments(segments: Iterable[Any]) -> Iterator[Word]:
    """Yield every timed word across all segments as one continuous stream."""
    for segment in segments:
        words = getattr(segment, "words", None)
        if words:
            for word in words:
                text = str(getattr(word, "word", "")).strip()
                if text:
                    yield Word(text, float(word.start), float(word.end))
            continue
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        yield from _spread_text(text, float(segment.start), float(segment.end))


def _spread_text(text: str, start: float, end: float) -> Iterator[Word]:
    """Spread untimed segment text evenly over the segment duration."""
    tokens = text.split()
    if not tokens:
        return
    span = max(end - start, 0.0)
    step = span / len(tokens) if tokens else 0.0
    for index, token in enumerate(tokens):
        token_start = start + step * index
        yield Word(token, token_start, token_start + step)


def _sanitize(words: Iterable[Word]) -> list[Word]:
    """Clamp negative times and force the word stream to move forward."""
    result: list[Word] = []
    previous_end = 0.0
    for word in words:
        start = max(0.0, word.start, previous_end)
        end = word.end if word.end > start else start + 0.02
        result.append(Word(word.text, start, end))
        previous_end = end
    return result


def _char_count(words: Sequence[Word]) -> int:
    """Visible length of the words when joined with single spaces."""
    if not words:
        return 0
    return sum(len(w.text) for w in words) + len(words) - 1


def _tokens(text: str) -> list[str]:
    """Split rendered cue text back into individual tokens."""
    return text.replace("\n", " ").split()


def _min_lines(tokens: Sequence[str], limit: int) -> int:
    """Smallest number of lines that holds the tokens within the limit."""
    lines = 1
    used = 0
    for token in tokens:
        if used and used + 1 + len(token) > limit:
            lines += 1
            used = len(token)
        else:
            used = used + 1 + len(token) if used else len(token)
    return lines


def _split_score(words: Sequence[Word], index: int) -> float:
    """Score a candidate split point. A higher score is a better break."""
    previous = words[index - 1].text
    following = words[index].text
    score = 0.0
    if _ends_sentence(previous):
        score += 100.0
    elif _ends_clause(previous):
        score += 60.0
    gap = words[index].start - words[index - 1].end
    if gap > 0:
        score += min(40.0, gap * 60.0)
    lowered = following.lower().strip(".,;:!?")
    if lowered in _NO_LINE_END:
        score += 20.0
    if _is_weak_line_end(previous):
        score -= 40.0
    return score


def _best_layout(tokens: Sequence[str], lines: int, limit: int) -> list[str]:
    """Choose the most readable way to break tokens over a fixed line count."""
    count = len(tokens)
    lengths = [len(t) for t in tokens]
    prefix = [0] * (count + 1)
    for i, length in enumerate(lengths):
        prefix[i + 1] = prefix[i] + length + 1
    target = (prefix[count] - 1) / lines

    def line_length(start: int, stop: int) -> int:
        return prefix[stop] - prefix[start] - 1

    best_cost = float("inf")
    best_cuts: list[int] = []

    def walk(start: int, remaining: int, cuts: list[int], cost: float) -> None:
        nonlocal best_cost, best_cuts
        if cost >= best_cost:
            return
        if remaining == 1:
            length = line_length(start, count)
            total = cost + _line_cost(tokens, start, count, count, length, target, limit, lines)
            if total < best_cost:
                best_cost = total
                best_cuts = [*cuts, count]
            return
        for stop in range(start + 1, count - remaining + 2):
            length = line_length(start, stop)
            if length > limit and stop > start + 1:
                break
            walk(
                stop,
                remaining - 1,
                [*cuts, stop],
                cost + _line_cost(tokens, start, stop, count, length, target, limit, lines),
            )

    walk(0, lines, [], 0.0)
    if not best_cuts:
        return [" ".join(tokens)]
    layout: list[str] = []
    previous = 0
    for cut in best_cuts:
        layout.append(" ".join(tokens[previous:cut]))
        previous = cut
    return [line for line in layout if line]


def _line_cost(
    tokens: Sequence[str],
    start: int,
    stop: int,
    count: int,
    length: int,
    target: float,
    limit: int,
    lines: int,
) -> float:
    """Cost of placing ``tokens[start:stop]`` on one line."""
    cost = (length - target) ** 2
    if length > limit:
        cost += (length - limit) * 1000.0
    if stop < count:
        last = tokens[stop - 1]
        if _is_weak_line_end(last):
            cost += 1000.0
        elif _ends_sentence(last) or _ends_clause(last):
            cost -= 30.0
        if (stop - start) < 3 and count >= 6:
            cost += 1000.0
        # Prefer the bottom-heavy pyramid: an earlier line at least as long.
        cost -= min(length, limit) * 0.2 * (lines - 1)
    return cost


def sidecar_path(video: Path, language_code: str = "en", flags: Sequence[str] = ()) -> Path:
    """Build the sidecar subtitle path that Plex and Jellyfin recognize.

    Args:
        video: The source video file.
        language_code: An ISO 639-1 or ISO 639-2 code, for example ``en``.
        flags: Extra markers such as ``forced`` or ``sdh``.

    Returns:
        The sidecar path next to the video, for example
        ``Movie.2019.1080p.BluRay.en.srt``.

    Raises:
        ValueError: If the language code is not a plain language tag.
    """
    if not _LANGUAGE_CODE_RE.match(language_code):
        raise ValueError(f"Invalid subtitle language code: {language_code!r}")
    base = video.stem if video.suffix.lower() in VIDEO_EXTENSIONS else video.name
    return video.with_name(".".join([base, language_code, *flags, "srt"]))


def _write_all(fd: int, payload: bytes) -> None:
    """Write every byte, because one os.write call may write only part of it.

    Args:
        fd: An open file descriptor.
        payload: The bytes to write.

    Raises:
        OSError: If the descriptor stops accepting data before the end.
    """
    written = 0
    while written < len(payload):
        count = os.write(fd, payload[written:])
        if count <= 0:
            raise OSError(f"Wrote only {written} of {len(payload)} bytes")
        written += count


def _temp_path(source_video: Path) -> Path:
    """Build a hidden temporary path next to the video.

    The uuid adds 38 characters to the name. Most filesystems cap a name at 255
    bytes, so the stem is trimmed to leave room rather than failing with
    ``ENAMETOOLONG`` on a long release name.
    """
    suffix = f".{uuid.uuid4().hex}.srt.tmp"
    budget = _MAX_FILENAME_BYTES - len(suffix.encode()) - 1
    stem = source_video.stem
    while len(stem.encode()) > budget and stem:
        stem = stem[:-1]
    return source_video.with_name(f".{stem}{suffix}")


class AtomicWriter:
    """Writes subtitle files with a crash-safe rename and inherited ownership."""

    @staticmethod
    def write_srt(source_video: Path, srt_content: str, language_code: str = "en") -> Path:
        """Write the SRT next to the video and commit it with a single rename.

        The content goes to a hidden temporary file in the same directory, is
        flushed to disk, and only then replaces the final name. A reader
        therefore sees either no file or a complete file.

        Args:
            source_video: The source video file.
            srt_content: The finished SRT document.
            language_code: The language suffix for the sidecar name.

        Returns:
            The path of the committed subtitle file.

        Raises:
            RuntimeError: If the write or the rename fails.
        """
        final_path = sidecar_path(source_video, language_code)
        temp_path = _temp_path(source_video)

        logger.debug(f"Writing subtitle to temporary file {temp_path.name}")
        payload = srt_content.encode("utf-8")

        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                _write_all(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)

            AtomicWriter._inherit_metadata(source_video, temp_path)
        except Exception as error:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Atomic subtitle write failed for {source_video.name}: {error}"
            ) from error

        try:
            os.replace(temp_path, final_path)
        except OSError as error:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not commit the subtitle for {source_video.name}: {error}"
            ) from error

        # The file is committed. A failure past this point is not a write failure,
        # so it must not be reported as one.
        AtomicWriter._sync_directory(final_path.parent)
        logger.info(f"Wrote {final_path.name}")
        return final_path

    @staticmethod
    def _inherit_metadata(source_video: Path, temp_path: Path) -> None:
        """Copy owner and permission bits from the video to the subtitle."""
        try:
            stat = source_video.stat()
        except OSError as error:
            logger.debug(f"Cannot read metadata of {source_video.name}: {error}")
            return

        try:
            os.chown(temp_path, stat.st_uid, stat.st_gid)
        except (OSError, AttributeError) as error:
            # chown needs root on most systems, and SMB/NFS mounts often deny it.
            logger.debug(f"Cannot set owner on {temp_path.name}: {error}")

        try:
            os.chmod(temp_path, stat.st_mode & 0o666)
        except OSError as error:
            logger.debug(f"Cannot set permissions on {temp_path.name}: {error}")

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        """Flush the directory entry so the rename survives a power loss."""
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
