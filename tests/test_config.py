"""Tests for the configuration schemas."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aisrt.config import AppConfig, FilterConfig, HardwareConfig


def test_hardware_config_defaults() -> None:
    """Test HardwareConfig default values."""
    config = HardwareConfig()
    assert config.force_device is None
    assert config.force_compute_type is None
    assert config.force_model is None


def test_filter_config_defaults() -> None:
    """Test FilterConfig default values."""
    config = FilterConfig()
    assert config.min_age_mins == 15
    assert ".mkv" in config.extensions
    assert "eng" in config.target_languages


def test_app_config_defaults(tmp_path: Path) -> None:
    """Test AppConfig initialization and defaults."""
    media_dir = tmp_path / "media"
    config = AppConfig(media_dir=media_dir)
    assert config.media_dir == media_dir
    assert config.dry_run is False
    assert config.db_path == Path.home() / ".config" / "aisrt" / "state.db"
    assert isinstance(config.hardware, HardwareConfig)
    assert isinstance(config.filters, FilterConfig)


def test_app_config_validation_error() -> None:
    """Test AppConfig fails without required media_dir."""
    with pytest.raises(ValidationError):
        AppConfig()  # type: ignore[call-arg]
