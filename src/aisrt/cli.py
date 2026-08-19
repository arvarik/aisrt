"""Command line interface for the SRT generator."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from aisrt import __version__
from aisrt.assembly import SRTFormatter, SubtitleStyle
from aisrt.config import AppConfig, SubtitleConfig
from aisrt.discovery import ACTION_PROCESS, DiscoveryEngine
from aisrt.hardware import HardwareProfiler, ModelConfig, ModelRouter
from aisrt.pipeline import Pipeline, PipelineStats, ProgressReporter
from aisrt.probing import FFmpegNotFoundError, require_ffmpeg
from aisrt.state import StateTracker
from aisrt.stt import STTWorker

app = typer.Typer(help="Generate broadcast-quality subtitles for a media library.")
console = Console()

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130


def configure_logging(verbose: bool) -> None:
    """Send log records to stderr so they never disturb the progress display."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}",
    )


def configure_threading(model_config: ModelConfig) -> None:
    """Set the OpenMP thread limits that CTranslate2 reads at load time.

    CTranslate2 passes ``cpu_threads`` straight to its own intra-operation pool,
    so pinning OpenMP to one thread would cripple CPU inference. A value the user
    already set is left alone.

    Args:
        model_config: The resolved model settings.
    """
    threads = "1" if model_config.device == "cuda" else str(max(1, model_config.cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, threads)


def _drop_unset(values: dict[str, Any]) -> dict[str, Any]:
    """Remove the options the user did not type.

    Every CLI option defaults to None. Filtering the unset ones out before the
    settings object is built is what lets an ``AISRT_*`` environment variable
    take effect.
    """
    return {key: value for key, value in values.items() if value is not None}


def build_config(
    media_dir: Path,
    *,
    min_age_mins: int | None = None,
    extensions: list[str] | None = None,
    exclude: list[str] | None = None,
    languages: list[str] | None = None,
    db_path: Path | None = None,
    dry_run: bool | None = None,
    translate: bool | None = None,
    language: str | None = None,
    watch: bool | None = None,
    watch_interval: int | None = None,
    max_memory_mb: int | None = None,
    force_device: str | None = None,
    force_model: str | None = None,
    force_compute_type: str | None = None,
    batch_size: int | None = None,
    max_chars_per_line: int | None = None,
    max_cps: float | None = None,
) -> AppConfig:
    """Assemble the configuration from the options the user typed.

    Args:
        media_dir: The directory to scan.
        min_age_mins: Skip files modified within this many minutes.
        extensions: Media extensions to accept.
        exclude: Glob patterns to skip.
        languages: Subtitle languages to generate.
        db_path: Path of the state database.
        dry_run: Report without running inference.
        translate: Translate speech into English.
        language: Force the spoken language.
        watch: Keep running and rescan on an interval.
        watch_interval: Minutes between scans.
        max_memory_mb: Cap on decoded audio held in memory.
        force_device: Compute device to use.
        force_model: Model name or local model directory.
        force_compute_type: Compute precision to use.
        batch_size: Batch size for batched inference.
        max_chars_per_line: Maximum characters on a subtitle line.
        max_cps: Maximum reading speed in characters per second.

    Returns:
        The validated configuration.
    """
    hardware = _drop_unset(
        {
            "force_device": force_device,
            "force_model": force_model,
            "force_compute_type": force_compute_type,
            "batch_size": batch_size,
            "prefer_accuracy": False if batch_size else None,
        }
    )
    filters = _drop_unset(
        {
            "min_age_mins": min_age_mins,
            "extensions": extensions or None,
            "exclude_patterns": exclude or None,
            "target_languages": languages or None,
        }
    )
    subtitles = _drop_unset({"max_chars_per_line": max_chars_per_line, "max_cps": max_cps})
    top_level = _drop_unset(
        {
            "db_path": db_path,
            "dry_run": dry_run,
            "translate": translate,
            "language": language,
            "watch": watch,
            "watch_interval_mins": watch_interval,
            "max_memory_mb": max_memory_mb,
        }
    )
    # Pass mappings, not model instances. pydantic-settings merges a mapping
    # with the matching environment sub-tree, while a constructed model replaces
    # it wholesale and would silence every AISRT_HARDWARE__* variable the moment
    # one hardware option is typed.
    if hardware:
        top_level["hardware"] = hardware
    if filters:
        top_level["filters"] = filters
    if subtitles:
        top_level["subtitles"] = subtitles
    return AppConfig(media_dir=media_dir, **top_level)


def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        # Rich would colour and highlight this, which breaks a script parsing it.
        console.print(f"aisrt {__version__}", highlight=False)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit",
        ),
    ] = False,
) -> None:
    """Generate broadcast-quality subtitles for a media library."""


@app.command()
def scan(
    media_dir: Annotated[Path, typer.Argument(help="Root directory containing media files")],
    min_age_mins: Annotated[
        int | None, typer.Option(help="Skip files modified within this many minutes")
    ] = None,
    ext: Annotated[
        list[str] | None, typer.Option("--ext", help="Media extension to accept (repeatable)")
    ] = None,
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", help="Glob pattern to skip (repeatable)")
    ] = None,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Subtitle language (repeatable)")
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Path of the state database")] = None,
    limit: Annotated[int, typer.Option(help="Stop after this many rows in the report")] = 200,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
) -> None:
    """Report what a run would do, without transcribing anything."""
    configure_logging(verbose)
    try:
        config = build_config(
            media_dir,
            min_age_mins=min_age_mins,
            extensions=ext,
            exclude=exclude,
            languages=lang,
            db_path=db_path,
            dry_run=True,
        )
        config.require_media_dir()
        require_ffmpeg()
    except (ValueError, FFmpegNotFoundError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        raise typer.Exit(EXIT_CONFIG) from error

    console.print("\n[bold cyan]Hardware[/bold cyan]")
    profile = HardwareProfiler.profile()
    ModelRouter.get_config(profile, config.hardware, translate=config.translate)

    console.print(f"\n[bold cyan]Scanning {config.media_dir}[/bold cyan]")
    raise typer.Exit(asyncio.run(_run_scan(config, limit)))


async def _run_scan(config: AppConfig, limit: int) -> int:
    """Crawl the library and print a report.

    Args:
        config: The validated configuration.
        limit: The most rows to show in the table.

    Returns:
        The process exit code.
    """
    table = Table(title="Discovery report", show_lines=False)
    table.add_column("File", style="dim", no_wrap=False)
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Action", style="magenta")
    table.add_column("Reason", style="yellow")

    process_count = 0
    skip_count = 0
    truncated = 0

    async with StateTracker(config.db_path) as tracker:
        engine = DiscoveryEngine(config.media_dir, config.filters, tracker)
        with _spinner() as progress:
            task = progress.add_task("Scanning...", total=None)
            async for media_file, action in engine.scan():
                size_mb = media_file.size / (1024 * 1024)
                try:
                    label = str(media_file.path.relative_to(config.media_dir))
                except ValueError:
                    label = str(media_file.path)

                if action == ACTION_PROCESS:
                    process_count += 1
                    verdict, reason = "[bold green]PROCESS[/bold green]", "Needs a subtitle"
                else:
                    skip_count += 1
                    verdict, reason = "[dim]SKIP[/dim]", action.replace("SKIP: ", "")
                row = (label, f"{size_mb:.1f}", verdict, reason)

                if process_count + skip_count <= limit:
                    table.add_row(*row)
                else:
                    truncated += 1
                progress.update(task, description=f"Scanned {process_count + skip_count} files")

    console.print(table)
    if truncated:
        console.print(f"[dim]{truncated} more file(s) not shown. Raise --limit to see them.[/dim]")
    console.print(f"\n[bold]{process_count}[/bold] to process, [bold]{skip_count}[/bold] skipped.")
    return EXIT_OK


@app.command()
def run(
    media_dir: Annotated[Path, typer.Argument(help="Root directory containing media files")],
    min_age_mins: Annotated[
        int | None, typer.Option(help="Skip files modified within this many minutes")
    ] = None,
    translate: Annotated[
        bool | None, typer.Option("--translate/--no-translate", help="Translate speech to English")
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", help="Force the spoken language, e.g. 'ja'")
    ] = None,
    watch: Annotated[
        bool | None, typer.Option("--watch/--no-watch", help="Keep running and rescan")
    ] = None,
    watch_interval: Annotated[
        int | None, typer.Option("--watch-interval", help="Minutes between scans in watch mode")
    ] = None,
    ext: Annotated[
        list[str] | None, typer.Option("--ext", help="Media extension to accept (repeatable)")
    ] = None,
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", help="Glob pattern to skip (repeatable)")
    ] = None,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Subtitle language (repeatable)")
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Path of the state database")] = None,
    max_memory_mb: Annotated[
        int | None, typer.Option(help="Cap on decoded audio held in memory, in megabytes")
    ] = None,
    force_device: Annotated[
        str | None, typer.Option(help="Compute device: cuda, cpu, or auto")
    ] = None,
    force_model: Annotated[
        str | None, typer.Option(help="Model name or a local model directory")
    ] = None,
    force_compute_type: Annotated[
        str | None, typer.Option(help="Compute precision: float16, int8_float16, or int8")
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Decode this many chunks together. Faster, slightly less accurate."),
    ] = None,
    max_chars_per_line: Annotated[
        int | None, typer.Option(help="Maximum characters on one subtitle line")
    ] = None,
    max_cps: Annotated[
        float | None, typer.Option(help="Maximum reading speed in characters per second")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
) -> None:
    """Transcribe every media file that is missing a subtitle."""
    configure_logging(verbose)
    try:
        config = build_config(
            media_dir,
            min_age_mins=min_age_mins,
            extensions=ext,
            exclude=exclude,
            languages=lang,
            db_path=db_path,
            translate=translate,
            language=language,
            watch=watch,
            watch_interval=watch_interval,
            max_memory_mb=max_memory_mb,
            force_device=force_device,
            force_model=force_model,
            force_compute_type=force_compute_type,
            batch_size=batch_size,
            max_chars_per_line=max_chars_per_line,
            max_cps=max_cps,
        )
        config.require_media_dir()
        require_ffmpeg()
    except (ValueError, FFmpegNotFoundError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        raise typer.Exit(EXIT_CONFIG) from error

    console.print("\n[bold cyan]Hardware[/bold cyan]")
    profile = HardwareProfiler.profile()
    model_config = ModelRouter.get_config(profile, config.hardware, translate=config.translate)
    configure_threading(model_config)

    console.print(f"\n[bold cyan]Processing {config.media_dir}[/bold cyan]")
    with STTWorker(model_config) as stt_worker:
        raise typer.Exit(asyncio.run(_execute(config, stt_worker)))


async def _execute(config: AppConfig, stt_worker: STTWorker) -> int:
    """Run the pipeline once, or repeatedly in watch mode.

    Args:
        config: The validated configuration.
        stt_worker: The loaded model.

    Returns:
        The process exit code.
    """
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    formatter = SRTFormatter(style=_style_from(config.subtitles))
    exit_code = EXIT_OK

    async with StateTracker(config.db_path) as tracker:
        while True:
            engine = DiscoveryEngine(config.media_dir, config.filters, tracker)
            reporter = ProgressReporter()
            pipeline = Pipeline(
                engine,
                stt_worker,
                formatter=formatter,
                translate=config.translate,
                language=config.language,
                max_memory_mb=config.max_memory_mb,
                extract_timeout_secs=config.extract_timeout_secs,
                stop_event=stop_event,
                progress=reporter,
            )
            try:
                async with _live_progress(reporter):
                    stats = await pipeline.run()
            except BaseExceptionGroup as group:
                # A TaskGroup reports every worker failure together.
                for error in group.exceptions:
                    logger.error(f"Pipeline error: {error}")
                return EXIT_FAILURES
            except asyncio.CancelledError:
                console.print("\n[bold red]Aborted.[/bold red]")
                return EXIT_INTERRUPTED

            _print_summary(stats)
            if stats.had_failures:
                exit_code = EXIT_FAILURES

            if stop_event.is_set():
                console.print("\n[bold yellow]Shutdown complete.[/bold yellow]")
                return EXIT_INTERRUPTED
            if not config.watch:
                return exit_code

            console.print(f"\n[dim]Sleeping {config.watch_interval_mins} minute(s).[/dim]")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=config.watch_interval_mins * 60)
            if stop_event.is_set():
                return EXIT_INTERRUPTED


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Ask the pipeline to drain on the first signal and abort on the second."""
    loop = asyncio.get_running_loop()
    state = {"count": 0}

    def handle(sig: signal.Signals) -> None:
        state["count"] += 1
        if state["count"] == 1:
            console.print(
                f"\n[bold yellow]{sig.name} received. Finishing the current file. "
                "Send it again to stop now.[/bold yellow]"
            )
            stop_event.set()
            return
        console.print("\n[bold red]Stopping now.[/bold red]")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, handle, sig)


@contextlib.asynccontextmanager
async def _live_progress(reporter: ProgressReporter) -> Any:
    """Render the transcription progress bar from the event loop thread."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task_id = progress.add_task("Waiting for the first file...", total=None)

    async def refresh() -> None:
        while True:
            if reporter.current_file:
                progress.update(
                    task_id,
                    description=f"Transcribing {reporter.current_file}",
                    total=reporter.total_seconds or None,
                    completed=reporter.completed_seconds,
                )
            else:
                progress.update(task_id, description="Extracting audio...", total=None)
            await asyncio.sleep(0.25)

    with progress:
        refresher = asyncio.create_task(refresh(), name="progress")
        try:
            yield
        finally:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher


def _spinner() -> Progress:
    """Build a lightweight spinner for the scan command."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def _style_from(subtitles: SubtitleConfig) -> SubtitleStyle:
    """Convert the subtitle settings into the formatter's style object."""
    return SubtitleStyle(
        max_chars_per_line=subtitles.max_chars_per_line,
        max_lines=subtitles.max_lines,
        max_cps=subtitles.max_cps,
        min_duration=subtitles.min_duration,
        max_duration=subtitles.max_duration,
        min_gap=subtitles.min_gap,
    )


def _print_summary(stats: PipelineStats) -> None:
    """Print the end-of-run dashboard."""
    hours = stats.total_audio_duration_secs / 3600
    message = (
        f"Scanned: [cyan]{stats.files_scanned}[/cyan]  "
        f"Processed: [green]{stats.files_processed}[/green]  "
        f"Skipped: [yellow]{stats.files_skipped}[/yellow]  "
        f"No speech: [yellow]{stats.files_without_speech}[/yellow]  "
        f"Failed: [red]{stats.files_failed}[/red]\n"
        f"Audio transcribed: [bold]{hours:.2f} hours[/bold]\n"
        f"Elapsed: [bold]{stats.elapsed:.1f} s[/bold] "
        f"([bold green]{stats.speedup:.1f}x[/bold green] real time)"
    )
    console.print(Panel(message, title="[bold]Run summary[/bold]", expand=False))


if __name__ == "__main__":
    app()
