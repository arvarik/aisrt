"""Shared fixtures and environment isolation for the test suite."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the developer's own AISRT_* variables from every test.

    Without this, a variable exported in the shell changes what the settings
    object resolves to and the config tests fail for reasons unrelated to the
    change under test.
    """
    for name in list(os.environ):
        if name.startswith("AISRT_") or name in {"HF_TOKEN", "XDG_CONFIG_HOME"}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def ffmpeg_installed() -> bool:
    """Report whether a real FFmpeg toolchain is installed."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass
class FakeWord:
    """Stand-in for a faster-whisper Word."""

    word: str
    start: float
    end: float


@dataclass
class FakeSegment:
    """Stand-in for a faster-whisper Segment."""

    text: str
    start: float
    end: float
    words: list[FakeWord] | None = field(default=None)


class SegmentFactory(Protocol):
    """Builds a fake segment with evenly spaced word timings."""

    def __call__(self, text: str, start: float = 0.0, seconds_per_word: float = 0.4) -> FakeSegment:
        """Build the segment."""
        ...


@pytest.fixture
def make_segment() -> SegmentFactory:
    """Return a helper that builds a segment with evenly spaced word timings."""

    def build(text: str, start: float = 0.0, seconds_per_word: float = 0.4) -> FakeSegment:
        words = []
        cursor = start
        for token in text.split():
            words.append(FakeWord(token, cursor, cursor + seconds_per_word))
            cursor += seconds_per_word
        return FakeSegment(text=text, start=start, end=cursor, words=words)

    return build
