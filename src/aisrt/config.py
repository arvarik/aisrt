"""Configuration schemas for the SRT Generator."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareConfig(BaseModel):
    """Configuration for hardware acceleration and inference limits."""

    force_device: str | None = Field(
        default=None,
        description="Force a specific compute device (e.g., 'cuda'). Auto-detect if None.",
    )
    force_compute_type: str | None = Field(
        default=None,
        description="Force compute type (e.g., 'float16', 'int8'). Auto-detect if None.",
    )
    force_model: str | None = Field(
        default=None,
        description="Force a Whisper model (e.g., 'large-v3-turbo'). Auto-detect if None.",
    )


class FilterConfig(BaseModel):
    """Configuration for filtering media files during discovery."""

    min_age_mins: int = Field(
        default=15,
        description="Minimum file age in minutes to avoid processing active downloads.",
    )
    extensions: list[str] = Field(
        default_factory=lambda: [".mkv", ".mp4", ".avi", ".webm"],
        description="List of valid media file extensions to process.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*sample*", "*extras*", "*featurettes*"],
        description="Glob patterns for directories or files to ignore.",
    )
    target_languages: list[str] = Field(
        default_factory=lambda: ["eng", "en"],
        description="Target subtitle languages to generate/check for.",
    )


class AppConfig(BaseSettings):
    """Main application configuration."""

    media_dir: Path = Field(
        description="The root directory containing media to scan.",
    )
    db_path: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "aisrt" / "state.db",
        description="Path to the local SQLite state database.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, only scan and report what would be done (no execution).",
    )
    translate: bool = Field(
        default=False,
        description="If True, translates foreign audio to English using Whisper's translate task.",
    )
    watch: bool = Field(
        default=False,
        description="If True, runs the pipeline continuously in daemon mode.",
    )
    watch_interval_mins: int = Field(
        default=60,
        description="Interval in minutes between scans when running in watch mode.",
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)

    model_config = SettingsConfigDict(
        env_prefix="AISRT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
