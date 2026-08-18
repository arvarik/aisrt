"""Tests for the command line interface."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from aisrt import __version__
from aisrt.cli import EXIT_CONFIG, EXIT_FAILURES, EXIT_OK, app, build_config, configure_threading
from aisrt.hardware import ModelConfig
from aisrt.pipeline import PipelineStats

runner = CliRunner()


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """Create an empty media directory."""
    media = tmp_path / "media"
    media.mkdir()
    return media


@pytest.fixture
def with_ffmpeg() -> Iterator[None]:
    """Pretend the FFmpeg toolchain is installed."""
    with patch("aisrt.cli.require_ffmpeg"):
        yield


class TestTopLevel:
    """The entry point must be discoverable and self-describing."""

    def test_help(self) -> None:
        """Both commands appear in the help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        assert "scan" in result.stdout
        assert "run" in result.stdout

    def test_version(self) -> None:
        """The reported version comes from the package, not a hardcoded string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == EXIT_OK
        assert __version__ in result.stdout
        assert "0.1.0" not in result.stdout or __version__ == "0.1.0"

    @pytest.mark.parametrize("command", ["scan", "run"])
    def test_command_help(self, command: str) -> None:
        """Each command documents its options."""
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == EXIT_OK
        assert "--verbose" in result.stdout


class TestDocumentedOptions:
    """Every option the documentation promises must exist."""

    @pytest.mark.parametrize(
        "option",
        [
            "--translate",
            "--watch",
            "--watch-interval",
            "--force-device",
            "--force-model",
            "--force-compute-type",
            "--min-age-mins",
            "--language",
            "--db-path",
            "--max-memory-mb",
            "--batch-size",
            "--ext",
            "--exclude",
            "--lang",
        ],
    )
    def test_run_options(self, option: str) -> None:
        """The run command accepts the documented option."""
        result = runner.invoke(app, ["run", "--help"])
        assert (
            option in result.stdout.replace("\n", "").replace(" ", "").replace("│", "")
            or option in result.stdout
        )

    @pytest.mark.parametrize("option", ["--min-age-mins", "--ext", "--exclude", "--db-path"])
    def test_scan_options(self, option: str) -> None:
        """The scan command accepts the documented option."""
        result = runner.invoke(app, ["scan", "--help"])
        assert option in result.stdout


class TestConfigurationErrors:
    """A bad invocation must fail fast, with a clear message."""

    def test_a_missing_directory(self) -> None:
        """The exit code marks a configuration problem."""
        result = runner.invoke(app, ["scan", "/definitely/not/here"])
        assert result.exit_code == EXIT_CONFIG
        assert "does not exist" in result.stdout

    def test_a_missing_ffmpeg(self, library: Path) -> None:
        """The message tells the user what to install."""
        from aisrt.probing import FFmpegNotFoundError

        with patch(
            "aisrt.cli.require_ffmpeg",
            side_effect=FFmpegNotFoundError("ffmpeg not found on PATH."),
        ):
            result = runner.invoke(app, ["scan", str(library)])
        assert result.exit_code == EXIT_CONFIG
        assert "ffmpeg not found" in result.stdout

    def test_an_invalid_option_value(self, library: Path, with_ffmpeg: None) -> None:
        """Validation runs before any hardware is touched."""
        result = runner.invoke(app, ["run", str(library), "--watch-interval", "0"])
        assert result.exit_code == EXIT_CONFIG


class TestBuildConfig:
    """The CLI must only pass the options the user actually typed."""

    def test_unset_options_leave_the_environment_in_charge(
        self, monkeypatch: pytest.MonkeyPatch, library: Path
    ) -> None:
        """This is what makes the documented AISRT_* variables work."""
        monkeypatch.setenv("AISRT_TRANSLATE", "true")
        monkeypatch.setenv("AISRT_WATCH_INTERVAL_MINS", "23")
        monkeypatch.setenv("AISRT_FILTERS__MIN_AGE_MINS", "77")

        config = build_config(library)

        assert config.translate is True
        assert config.watch_interval_mins == 23
        assert config.filters.min_age_mins == 77

    def test_a_typed_option_wins(self, monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
        """An explicit option overrides the environment."""
        monkeypatch.setenv("AISRT_TRANSLATE", "true")
        assert build_config(library, translate=False).translate is False

    def test_nested_overrides_reach_their_model(self, library: Path) -> None:
        """Hardware, filter, and subtitle options land in the right section."""
        config = build_config(
            library,
            force_device="cpu",
            force_model="tiny.en",
            force_compute_type="int8",
            min_age_mins=5,
            languages=["fr"],
            max_chars_per_line=37,
        )
        assert config.hardware.force_device == "cpu"
        assert config.hardware.force_model == "tiny.en"
        assert config.hardware.force_compute_type == "int8"
        assert config.filters.min_age_mins == 5
        assert config.filters.target_languages == ["fr"]
        assert config.subtitles.max_chars_per_line == 37

    def test_a_batch_size_turns_batching_on(self, library: Path) -> None:
        """Asking for throughput switches off the accuracy preference."""
        config = build_config(library, batch_size=8)
        assert config.hardware.batch_size == 8
        assert config.hardware.prefer_accuracy is False

    def test_the_default_prefers_accuracy(self, library: Path) -> None:
        """Without a batch size, sequential decoding stays on."""
        assert build_config(library).hardware.prefer_accuracy is True


class TestThreadConfiguration:
    """Thread limits must help CPU inference, not cripple it."""

    def test_the_cpu_path_gets_every_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CTranslate2 runs its own pool, so pinning OpenMP to one would hurt."""
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(name, raising=False)

        configure_threading(ModelConfig("tiny.en", "cpu", "int8", cpu_threads=8))
        assert os.environ["OMP_NUM_THREADS"] == "8"

    def test_the_gpu_path_keeps_one_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a GPU the CPU only prepares features, so one thread is enough."""
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(name, raising=False)

        configure_threading(ModelConfig("large-v3", "cuda", "float16", cpu_threads=4))
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_a_user_value_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator who tuned the environment keeps their setting."""
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        configure_threading(ModelConfig("tiny.en", "cpu", "int8", cpu_threads=8))
        assert os.environ["OMP_NUM_THREADS"] == "3"


class TestScanCommand:
    """The scan command reports without transcribing."""

    def test_an_empty_library(self, library: Path, with_ffmpeg: None) -> None:
        """A directory with nothing in it reports zero of each."""
        result = runner.invoke(app, ["scan", str(library)])
        assert result.exit_code == EXIT_OK
        assert "0" in result.stdout

    def test_reports_files_that_need_a_subtitle(self, library: Path, with_ffmpeg: None) -> None:
        """A discovered file appears in the report as PROCESS."""
        video = library / "movie.mkv"
        video.write_text("data")
        os.utime(video, (1_700_000_000, 1_700_000_000))

        info = MagicMock(probe_failed=False, audio_tracks=(MagicMock(),), duration=60.0)
        info.has_text_subtitle.return_value = False

        with patch("aisrt.discovery.probe_media", new=AsyncMock(return_value=info)):
            result = runner.invoke(app, ["scan", str(library)])

        assert result.exit_code == EXIT_OK
        assert "PROCESS" in result.stdout
        assert "movie.mkv" in result.stdout

    def test_a_relative_path_is_accepted(self, library: Path, with_ffmpeg: None) -> None:
        """Running from inside the library must not raise a path error."""
        original = os.getcwd()
        os.chdir(library)
        try:
            result = runner.invoke(app, ["scan", "."])
        finally:
            os.chdir(original)
        assert result.exit_code == EXIT_OK


class TestRunCommand:
    """The run command must report the truth in its exit code."""

    def test_a_clean_run_exits_zero(self, library: Path) -> None:
        """Success is reported as success."""
        stats = PipelineStats(files_scanned=2, files_processed=2, start_time=0.0, end_time=1.0)
        pipeline = MagicMock()
        pipeline.run = AsyncMock(return_value=stats)
        with (
            patch("aisrt.cli.require_ffmpeg"),
            patch("aisrt.cli.STTWorker"),
            patch("aisrt.cli.Pipeline", return_value=pipeline),
        ):
            result = runner.invoke(app, ["run", str(library)])
        assert result.exit_code == EXIT_OK
        pipeline.run.assert_awaited_once()

    def test_a_failed_file_exits_nonzero(self, library: Path) -> None:
        """A cron job must be able to notice that files failed."""
        stats = PipelineStats(
            files_scanned=2, files_processed=1, files_failed=1, start_time=0.0, end_time=1.0
        )
        pipeline = MagicMock()
        pipeline.run = AsyncMock(return_value=stats)
        with (
            patch("aisrt.cli.require_ffmpeg"),
            patch("aisrt.cli.STTWorker"),
            patch("aisrt.cli.Pipeline", return_value=pipeline),
        ):
            result = runner.invoke(app, ["run", str(library)])
        assert result.exit_code == EXIT_FAILURES

    def test_the_summary_is_printed(self, library: Path) -> None:
        """The dashboard reports every counter."""
        stats = PipelineStats(
            files_scanned=5,
            files_processed=3,
            files_skipped=1,
            files_failed=0,
            files_without_speech=1,
            start_time=0.0,
            end_time=10.0,
            total_audio_duration_secs=3600.0,
        )
        pipeline = MagicMock()
        pipeline.run = AsyncMock(return_value=stats)
        with (
            patch("aisrt.cli.require_ffmpeg"),
            patch("aisrt.cli.STTWorker"),
            patch("aisrt.cli.Pipeline", return_value=pipeline),
        ):
            result = runner.invoke(app, ["run", str(library)])

        assert "Run summary" in result.stdout
        assert "1.00 hours" in result.stdout

    def test_translation_routes_to_a_capable_model(self, library: Path) -> None:
        """A translate run must not be handed the turbo checkpoint."""
        from aisrt.hardware import HardwareProfile

        stats = PipelineStats(start_time=0.0, end_time=1.0)
        pipeline = MagicMock()
        pipeline.run = AsyncMock(return_value=stats)
        captured: list[ModelConfig] = []

        def record(model_config: ModelConfig) -> MagicMock:
            """Remember the resolved model settings the CLI chose."""
            captured.append(model_config)
            return MagicMock()

        with (
            patch("aisrt.cli.require_ffmpeg"),
            patch(
                "aisrt.cli.HardwareProfiler.profile",
                return_value=HardwareProfile(True, 6.0, 32.0, 8, False),
            ),
            patch("aisrt.cli.STTWorker", side_effect=record),
            patch("aisrt.cli.Pipeline", return_value=pipeline),
        ):
            runner.invoke(app, ["run", str(library), "--translate"])

        assert captured, "the worker was never constructed"
        assert "turbo" not in captured[0].model_name
