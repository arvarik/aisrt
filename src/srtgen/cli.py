"""CLI commands for the SRT Generator."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from srtgen.config import AppConfig, FilterConfig, HardwareConfig
from srtgen.discovery import DiscoveryEngine
from srtgen.hardware import HardwareProfiler, ModelRouter, setup_thread_safety
from srtgen.state import StateTracker

app = typer.Typer(help="Ultimate SRT Generator", add_completion=False)
console = Console()


def configure_logging(verbose: bool) -> None:
    """Configure Loguru to output cleanly via Rich."""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, colorize=True)


@app.command()
def scan(
    media_dir: Annotated[Path, typer.Argument(help="Root directory containing media files")],
    min_age_mins: Annotated[int, typer.Option(help="Minimum file age in minutes")] = 15,
    force_device: Annotated[str | None, typer.Option(help="Force specific device")] = None,
    force_model: Annotated[str | None, typer.Option(help="Force specific model")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
) -> None:
    """Perform a dry-run scan of the media directory and profile hardware."""
    configure_logging(verbose)

    # Compile the configuration
    hw_config = HardwareConfig(force_device=force_device, force_model=force_model)
    flt_config = FilterConfig(min_age_mins=min_age_mins)
    config = AppConfig(
        media_dir=media_dir,
        dry_run=True,
        hardware=hw_config,
        filters=flt_config,
    )

    # 1. Profile Hardware
    setup_thread_safety()
    console.print("\n[bold cyan]1. Profiling Hardware...[/bold cyan]")
    profile = HardwareProfiler.profile()
    _ = ModelRouter.get_config(profile, config.hardware)

    # 2. Run the Async Discovery Engine
    console.print(f"\n[bold cyan]2. Scanning Directory: {config.media_dir}[/bold cyan]")
    asyncio.run(_run_scan(config))


async def _run_scan(config: AppConfig) -> None:
    """Execute the asynchronous scanning process."""
    table = Table(title="Media File Discovery Report", show_lines=True)
    table.add_column("File Path", style="dim", no_wrap=False)
    table.add_column("Size (MB)", justify="right", style="green")
    table.add_column("Action", style="magenta")
    table.add_column("Reason", style="yellow")

    async with StateTracker(config.db_path) as tracker:
        engine = DiscoveryEngine(config.media_dir, config.filters, tracker)

        process_count = 0
        skip_count = 0

        async for media_file, action_str in engine.scan():
            size_mb = media_file.size / (1024 * 1024)
            path_str = str(media_file.path.relative_to(config.media_dir))

            if action_str == "PROCESS":
                table.add_row(
                    path_str,
                    f"{size_mb:.1f}",
                    "[bold green]PROCESS[/bold green]",
                    "Needs Subtitle",
                )
                process_count += 1
            else:
                reason = action_str.replace("SKIP: ", "")
                table.add_row(path_str, f"{size_mb:.1f}", "[dim]SKIP[/dim]", reason)
                skip_count += 1

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {process_count} files to process, {skip_count} files skipped."
    )


@app.command()
def run(
    media_dir: Annotated[Path, typer.Argument(help="Root directory containing media files")],
    min_age_mins: Annotated[int, typer.Option(help="Minimum file age in minutes")] = 15,
    translate: Annotated[
        bool, typer.Option("--translate", help="Enable AI translation to English")
    ] = False,
    watch: Annotated[bool, typer.Option("--watch", help="Run continuously in daemon mode")] = False,
    watch_interval: Annotated[
        int, typer.Option("--watch-interval", help="Minutes between scans in watch mode")
    ] = 60,
    force_device: Annotated[str | None, typer.Option(help="Force specific device")] = None,
    force_model: Annotated[str | None, typer.Option(help="Force specific model")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
) -> None:
    """Run the live SRT generation pipeline."""
    configure_logging(verbose)

    # Compile the configuration
    hw_config = HardwareConfig(force_device=force_device, force_model=force_model)
    flt_config = FilterConfig(min_age_mins=min_age_mins)
    config = AppConfig(
        media_dir=media_dir,
        dry_run=False,
        translate=translate,
        watch=watch,
        watch_interval_mins=watch_interval,
        hardware=hw_config,
        filters=flt_config,
    )

    setup_thread_safety()

    console.print("\n[bold cyan]1. Profiling Hardware & Initializing Models...[/bold cyan]")
    profile = HardwareProfiler.profile()
    model_cfg = ModelRouter.get_config(profile, config.hardware)

    # Initialize the STT singleton before starting async loop
    from srtgen.stt import STTWorker

    stt_worker = STTWorker()
    stt_worker.initialize(model_cfg)

    console.print(f"\n[bold cyan]2. Starting Async Pipeline on {config.media_dir}[/bold cyan]")

    # We define a wrapper to inject the db context manager and the pipeline
    async def _execute_pipeline() -> None:
        from srtgen.pipeline import Pipeline

        async with StateTracker(config.db_path) as tracker:
            while True:
                engine = DiscoveryEngine(config.media_dir, config.filters, tracker)
                pipeline = Pipeline(
                    engine, cpu_cores=profile.physical_cores, translate=config.translate
                )
                await pipeline.run()

                if not config.watch:
                    break

                console.print(
                    f"\n[bold yellow]Sleeping for {config.watch_interval_mins} "
                    f"minutes...[/bold yellow]"
                )
                await asyncio.sleep(config.watch_interval_mins * 60)
                console.print(
                    f"\n[bold cyan]Waking up and scanning Directory: {config.media_dir}[/bold cyan]"
                )

    try:
        asyncio.run(_execute_pipeline())
        console.print("\n[bold green]Pipeline finished successfully.[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline interrupted by user.[/bold red]")
    finally:
        stt_worker.close()


if __name__ == "__main__":
    app()
