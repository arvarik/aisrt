"""Configuration schemas for the SRT generator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_db_path() -> Path:
    """Return the default state database path.

    Honours ``XDG_CONFIG_HOME`` so that a container can place the database on a
    writable volume without running as root.
    """
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "aisrt" / "state.db"


class HardwareConfig(BaseModel):
    """Overrides for hardware detection and model routing."""

    force_device: str | None = Field(
        default=None,
        description="Compute device to use ('cuda', 'cpu', or 'auto'). Detect when None.",
    )
    force_compute_type: str | None = Field(
        default=None,
        description="Compute type ('float16', 'int8_float16', 'int8'). Detect when None.",
    )
    force_model: str | None = Field(
        default=None,
        description="Whisper model name or a local model directory. Route when None.",
    )
    batch_size: int = Field(
        default=0,
        ge=0,
        le=64,
        description="Batch size for batched inference. 0 lets the router decide.",
    )
    prefer_accuracy: bool = Field(
        default=True,
        description=(
            "Prefer sequential decoding, which keeps the temperature fallback and the "
            "hallucination guard that batched inference discards."
        ),
    )

    @field_validator("force_device")
    @classmethod
    def _check_device(cls, value: str | None) -> str | None:
        """Accept only a device CTranslate2 can use."""
        if value is None:
            return None
        normalized = value.strip().lower()
        allowed = {"cuda", "cpu", "auto"}
        if normalized not in allowed:
            raise ValueError(f"force_device must be one of {sorted(allowed)}, not {value!r}")
        return normalized


class FilterConfig(BaseModel):
    """Rules that decide which media files enter the pipeline."""

    min_age_mins: int = Field(
        default=15,
        ge=0,
        le=100_000,
        description="Skip files modified within this many minutes, to avoid active downloads.",
    )
    extensions: list[str] = Field(
        default_factory=lambda: [".mkv", ".mp4", ".avi", ".webm", ".ts", ".m2ts", ".vob"],
        min_length=1,
        description="Media file extensions to process.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*sample*", "*extras*", "*featurettes*", "*trailer*"],
        description="Glob patterns matched against each path component, ignoring case.",
    )
    target_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        min_length=1,
        description="Subtitle languages to generate and to look for.",
    )
    follow_symlinks: bool = Field(
        default=False,
        description="Descend into symlinked directories. Loops are detected and skipped.",
    )

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        """Lowercase each extension and give it a leading dot."""
        normalized = []
        for item in value:
            cleaned = item.strip().lower()
            if not cleaned:
                continue
            normalized.append(cleaned if cleaned.startswith(".") else f".{cleaned}")
        if not normalized:
            raise ValueError("extensions must contain at least one entry")
        return sorted(set(normalized))

    @field_validator("exclude_patterns")
    @classmethod
    def _normalize_patterns(cls, value: list[str]) -> list[str]:
        """Lowercase the patterns so matching can ignore case."""
        return [item.strip().lower() for item in value if item.strip()]

    @field_validator("target_languages")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        """Lowercase and deduplicate the language codes, keeping their order."""
        seen: list[str] = []
        for item in value:
            cleaned = item.strip().lower()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        if not seen:
            raise ValueError("target_languages must contain at least one entry")
        return seen


class SubtitleConfig(BaseModel):
    """Broadcast layout and timing limits for generated cues."""

    max_chars_per_line: int = Field(
        default=42, ge=20, le=100, description="Maximum characters on one subtitle line."
    )
    max_lines: int = Field(default=2, ge=1, le=3, description="Maximum lines in one cue.")
    max_cps: float = Field(
        default=20.0, gt=0, le=60.0, description="Maximum reading speed in characters per second."
    )
    min_duration: float = Field(
        default=5.0 / 6.0, gt=0, le=5.0, description="Minimum cue duration in seconds."
    )
    max_duration: float = Field(
        default=7.0, gt=0, le=30.0, description="Maximum cue duration in seconds."
    )
    min_gap: float = Field(
        default=0.084, ge=0, le=1.0, description="Minimum gap between cues in seconds."
    )

    @model_validator(mode="after")
    def _check_durations(self) -> SubtitleConfig:
        """Reject a minimum duration that exceeds the maximum."""
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration must not be greater than max_duration")
        return self


class AppConfig(BaseSettings):
    """The complete application configuration.

    Values come from, in decreasing priority: arguments passed to the
    constructor, ``AISRT_*`` environment variables, then the defaults. The CLI
    passes an option only when the user typed it, so an environment variable is
    used whenever the matching option is absent.
    """

    media_dir: Path = Field(description="The root directory to scan.")
    db_path: Path = Field(
        default_factory=default_db_path, description="Path of the SQLite state database."
    )
    dry_run: bool = Field(
        default=False, description="Report what would happen without running inference."
    )
    translate: bool = Field(
        default=False, description="Translate non-English audio into English subtitles."
    )
    language: str | None = Field(
        default=None,
        description="Force the spoken language, for example 'ja'. Detect it when None.",
    )
    watch: bool = Field(default=False, description="Keep running and rescan on an interval.")
    watch_interval_mins: int = Field(
        default=60, ge=1, le=100_000, description="Minutes between scans in watch mode."
    )
    max_memory_mb: int = Field(
        default=2048,
        ge=256,
        le=1_000_000,
        description="Cap on decoded audio held in memory at one time, in megabytes.",
    )
    extract_timeout_secs: float = Field(
        default=1800.0, gt=0, description="Seconds to wait for FFmpeg on one file."
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)

    model_config = SettingsConfigDict(
        env_prefix="AISRT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("media_dir", "db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: Any) -> Any:
        """Expand ``~`` and make the path absolute."""
        if isinstance(value, str | Path):
            return Path(value).expanduser().resolve()
        return value

    def require_media_dir(self) -> Path:
        """Return the media directory after checking that it is usable.

        Returns:
            The resolved directory.

        Raises:
            ValueError: If the path is missing or is not a directory.
        """
        if not self.media_dir.exists():
            raise ValueError(f"Media directory does not exist: {self.media_dir}")
        if not self.media_dir.is_dir():
            raise ValueError(f"Media path is not a directory: {self.media_dir}")
        return self.media_dir
