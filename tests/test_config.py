"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aisrt.config import (
    AppConfig,
    FilterConfig,
    HardwareConfig,
    SubtitleConfig,
    default_db_path,
)


class TestDefaults:
    """The defaults must be usable without any configuration."""

    def test_minimal_config(self, tmp_path: Path) -> None:
        """Only the media directory is required."""
        config = AppConfig(media_dir=tmp_path)
        assert config.media_dir == tmp_path.resolve()
        assert config.dry_run is False
        assert config.translate is False
        assert config.watch is False
        assert config.watch_interval_mins == 60

    def test_db_path_follows_xdg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A container can move the database with XDG_CONFIG_HOME."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        assert default_db_path() == tmp_path / "cfg" / "aisrt" / "state.db"

    def test_paths_are_resolved(self, tmp_path: Path) -> None:
        """A relative path becomes absolute, so database keys stay stable."""
        nested = tmp_path / "media"
        nested.mkdir()
        config = AppConfig(media_dir=Path(f"{nested}/../media"))
        assert config.media_dir == nested.resolve()
        assert config.media_dir.is_absolute()


class TestValidation:
    """Bad settings must fail at startup, not halfway through a run."""

    def test_a_zero_watch_interval_is_rejected(self, tmp_path: Path) -> None:
        """A zero interval would turn the daemon into a busy loop."""
        with pytest.raises(ValidationError, match="watch_interval_mins"):
            AppConfig(media_dir=tmp_path, watch_interval_mins=0)

    def test_a_negative_watch_interval_is_rejected(self, tmp_path: Path) -> None:
        """Time cannot run backwards."""
        with pytest.raises(ValidationError, match="watch_interval_mins"):
            AppConfig(media_dir=tmp_path, watch_interval_mins=-5)

    def test_a_negative_min_age_is_rejected(self) -> None:
        """A negative age would disable the active-download guard."""
        with pytest.raises(ValidationError, match="min_age_mins"):
            FilterConfig(min_age_mins=-1)

    def test_an_empty_extension_list_is_rejected(self) -> None:
        """A run with no extensions would silently find nothing."""
        with pytest.raises(ValidationError):
            FilterConfig(extensions=[])

    def test_an_unknown_device_is_rejected(self) -> None:
        """A typo must not reach CTranslate2 as an opaque error."""
        with pytest.raises(ValidationError, match="force_device"):
            HardwareConfig(force_device="gpu")

    def test_a_minimum_above_the_maximum_is_rejected(self) -> None:
        """The cue duration range must be usable."""
        with pytest.raises(ValidationError, match="min_duration"):
            SubtitleConfig(min_duration=10.0, max_duration=7.0)

    def test_a_missing_media_directory_is_reported(self, tmp_path: Path) -> None:
        """The message names the path, so a typo is obvious."""
        config = AppConfig(media_dir=tmp_path / "absent")
        with pytest.raises(ValueError, match="does not exist"):
            config.require_media_dir()

    def test_a_file_is_not_a_media_directory(self, tmp_path: Path) -> None:
        """Pointing at a file is a configuration error."""
        target = tmp_path / "movie.mkv"
        target.write_text("data")
        config = AppConfig(media_dir=target)
        with pytest.raises(ValueError, match="not a directory"):
            config.require_media_dir()


class TestNormalisation:
    """User input is cleaned up before it reaches the matching code."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (["mkv", "MP4"], [".mkv", ".mp4"]),
            ([".MKV", ".mkv"], [".mkv"]),
            (["  .avi  "], [".avi"]),
        ],
    )
    def test_extensions(self, raw: list[str], expected: list[str]) -> None:
        """Extensions gain a leading dot, lose their case, and deduplicate."""
        assert FilterConfig(extensions=raw).extensions == expected

    def test_exclude_patterns_are_lowercased(self) -> None:
        """Matching is case-insensitive, so the patterns are stored lowercase."""
        config = FilterConfig(exclude_patterns=["*Sample*", "*EXTRAS*"])
        assert config.exclude_patterns == ["*sample*", "*extras*"]

    def test_languages_keep_their_order(self) -> None:
        """The first language names the subtitle file."""
        config = FilterConfig(target_languages=["EN", "fr", "en"])
        assert config.target_languages == ["en", "fr"]


class TestEnvironment:
    """Documented AISRT_* variables must actually take effect."""

    def test_top_level_variables(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A variable is read when the matching option is not passed."""
        monkeypatch.setenv("AISRT_TRANSLATE", "true")
        monkeypatch.setenv("AISRT_WATCH", "true")
        monkeypatch.setenv("AISRT_WATCH_INTERVAL_MINS", "17")

        config = AppConfig(media_dir=tmp_path)

        assert config.translate is True
        assert config.watch is True
        assert config.watch_interval_mins == 17

    def test_nested_variables(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The double underscore reaches into a nested model."""
        monkeypatch.setenv("AISRT_FILTERS__MIN_AGE_MINS", "45")
        monkeypatch.setenv("AISRT_HARDWARE__FORCE_MODEL", "tiny")
        monkeypatch.setenv("AISRT_HARDWARE__FORCE_DEVICE", "cpu")

        config = AppConfig(media_dir=tmp_path)

        assert config.filters.min_age_mins == 45
        assert config.hardware.force_model == "tiny"
        assert config.hardware.force_device == "cpu"

    def test_an_explicit_value_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A typed option overrides the environment, which is the expected order."""
        monkeypatch.setenv("AISRT_WATCH_INTERVAL_MINS", "17")
        assert AppConfig(media_dir=tmp_path, watch_interval_mins=99).watch_interval_mins == 99

    def test_unknown_variables_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An unrelated AISRT_ variable must not crash the run."""
        monkeypatch.setenv("AISRT_NOT_A_SETTING", "value")
        AppConfig(media_dir=tmp_path)
