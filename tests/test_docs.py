"""Tests that keep the documentation honest.

The README once documented options and environment variables that the code
silently ignored. These tests read the README and check every claim against the
running program, so the same drift cannot happen again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aisrt.cli import app
from aisrt.config import AppConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

runner = CliRunner()


def _flatten(text: str) -> str:
    """Strip the whitespace and box drawing that Rich wraps help output in."""
    return re.sub(r"[\s│]+", "", text)


def _help(*args: str) -> str:
    """Return the flattened help text for a command."""
    return _flatten(runner.invoke(app, [*args, "--help"]).stdout)


def _documented_options() -> list[tuple[str, str, str]]:
    """Read the option table from the README."""
    pattern = r"^\| `(--[a-z-]+)`(?:, `-\w`)? \| (yes|no|—) \| (yes|no|—) \|"
    rows = re.findall(pattern, README.read_text(encoding="utf-8"), re.M)
    assert rows, "the README option table could not be parsed"
    return rows


def _documented_env_vars() -> list[str]:
    """Read the environment variable table from the README."""
    names = re.findall(r"^\| `(AISRT_[A-Z_]+)` \|", README.read_text(encoding="utf-8"), re.M)
    assert names, "the README environment variable table could not be parsed"
    return names


# Each variable maps to the value to set and the config field it must change.
ENV_PROBES: dict[str, tuple[str, str, object]] = {
    "AISRT_TRANSLATE": ("true", "translate", True),
    "AISRT_WATCH": ("true", "watch", True),
    "AISRT_WATCH_INTERVAL_MINS": ("31", "watch_interval_mins", 31),
    "AISRT_LANGUAGE": ("ja", "language", "ja"),
    "AISRT_MAX_MEMORY_MB": ("777", "max_memory_mb", 777),
    "AISRT_FILTERS__MIN_AGE_MINS": ("41", "filters.min_age_mins", 41),
    "AISRT_FILTERS__TARGET_LANGUAGES": ('["fr","de"]', "filters.target_languages", ["fr", "de"]),
    "AISRT_HARDWARE__FORCE_MODEL": ("tiny", "hardware.force_model", "tiny"),
    "AISRT_HARDWARE__FORCE_DEVICE": ("cpu", "hardware.force_device", "cpu"),
    "AISRT_HARDWARE__FORCE_COMPUTE_TYPE": ("int8", "hardware.force_compute_type", "int8"),
    "AISRT_SUBTITLES__MAX_CPS": ("15.5", "subtitles.max_cps", 15.5),
    # db_path is resolved to an absolute path, so it is checked separately.
    "AISRT_DB_PATH": ("", "", None),
}


def _read(config: AppConfig, dotted: str) -> object:
    """Read a possibly nested attribute by its dotted name."""
    value: object = config
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


class TestDocumentedOptions:
    """Every option in the README table must exist, and only where claimed."""

    @pytest.mark.parametrize(("option", "in_scan", "in_run"), _documented_options())
    def test_option_presence_matches_the_readme(
        self, option: str, in_scan: str, in_run: str
    ) -> None:
        """A documented option exists, and an option marked absent really is."""
        flat = _flatten(option)
        scan_help, run_help = _help("scan"), _help("run")

        if in_scan == "yes":
            assert flat in scan_help, f"{option} is documented for scan but does not exist"
        elif in_scan == "no":
            assert flat not in scan_help, f"{option} exists on scan but the README says it does not"

        if in_run == "yes":
            assert flat in run_help, f"{option} is documented for run but does not exist"
        elif in_run == "no":
            assert flat not in run_help, f"{option} exists on run but the README says it does not"

    def test_version_is_available(self) -> None:
        """The README documents --version at the top level."""
        assert "--version" in _help()


class TestDocumentedEnvironmentVariables:
    """Every AISRT_ variable in the README must change the configuration."""

    @pytest.mark.parametrize("name", _documented_env_vars())
    def test_variable_takes_effect(
        self, name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Setting the variable, with no matching option passed, changes the config."""
        assert name in ENV_PROBES, f"{name} is documented but this test does not probe it"
        value, field, expected = ENV_PROBES[name]

        if name == "AISRT_DB_PATH":
            target = tmp_path / "custom" / "state.db"
            monkeypatch.setenv(name, str(target))
            assert AppConfig(media_dir=tmp_path).db_path == target
            return

        monkeypatch.setenv(name, value)
        assert _read(AppConfig(media_dir=tmp_path), field) == expected


class TestDocumentedExitCodes:
    """The exit code table drives cron jobs, so it must be accurate."""

    def test_a_configuration_error_exits_two(self, tmp_path: Path) -> None:
        """Code 2 means the configuration is wrong."""
        result = runner.invoke(app, ["scan", str(tmp_path / "absent")])
        assert result.exit_code == 2


class TestArchitectureDoc:
    """The architecture document must describe the shipped code."""

    def test_no_stale_poetry_instructions(self) -> None:
        """The project uses uv and hatchling, never Poetry."""
        for name in ("README.md", "CONTRIBUTING.md", "docs/ARCHITECTURE.md"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8").lower()
            assert "poetry install" not in text, f"{name} still tells contributors to use Poetry"

    def test_changelog_exists(self) -> None:
        """The release workflow reads CHANGELOG.md and links to it."""
        assert (REPO_ROOT / "CHANGELOG.md").is_file()

    def test_community_health_files_exist(self) -> None:
        """These files are what make the project usable by other people."""
        for name in ("SECURITY.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "LICENSE"):
            assert (REPO_ROOT / name).is_file(), f"{name} is missing"


class TestPackaging:
    """The package must be typed and importable as it advertises."""

    def test_py_typed_marker_is_present(self) -> None:
        """Without this file a type checker discards every annotation."""
        from importlib.resources import files

        assert files("aisrt").joinpath("py.typed").is_file()

    def test_version_is_not_hardcoded(self) -> None:
        """The version comes from the tag history, not from a literal."""
        from aisrt import __version__

        assert __version__
        assert __version__ != "0.1.0"
