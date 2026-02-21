"""CLI entry point for the Attractor pipeline engine.

Provides ``run``, ``validate``, and ``resume`` sub-commands using
Click and Rich for output formatting.

Usage::

    attractor run pipeline.dot --model gpt-4o --verbose
    attractor validate pipeline.dot --strict
    attractor resume checkpoint.json --verbose
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.interviewer import CLIInterviewer
from attractor.pipeline.models import Checkpoint, PipelineContext
from attractor.pipeline.parser import parse_dot_file
from attractor.pipeline.state import latest_checkpoint
from attractor.pipeline.stylesheet import ModelStylesheet
from attractor.pipeline.validator import ValidationLevel, has_errors, validate_pipeline
from attractor.pipeline.handlers import create_default_registry

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(package_name="attractor")
def main() -> None:
    """Attractor — a non-interactive coding agent for software factories."""


@main.command()
@click.argument("pipeline_dot", type=click.Path(exists=True))
@click.option("--model", default=None, help="Default model for codergen nodes.")
@click.option("--provider", default=None, help="LLM provider name.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=".attractor/checkpoints",
    help="Directory for checkpoint files.",
)
def run(
    pipeline_dot: str,
    model: str | None,
    provider: str | None,
    verbose: bool,
    checkpoint_dir: str,
) -> None:
    """Execute a pipeline from a DOT file."""
    _setup_logging(verbose)

    try:
        pipeline = parse_dot_file(pipeline_dot)
    except Exception as exc:
        console.print(f"[red]Failed to parse pipeline:[/red] {exc}")
        raise SystemExit(1) from exc

    # Validate first
    findings = validate_pipeline(pipeline)
    if has_errors(findings):
        console.print("[red]Pipeline validation failed:[/red]")
        for f in findings:
            console.print(f"  {f}")
        raise SystemExit(1)

    for f in findings:
        if f.level == ValidationLevel.WARNING:
            console.print(f"[yellow]Warning:[/yellow] {f}")

    # Build stylesheet from CLI options
    stylesheet_data: dict = {"rules": []}
    if model:
        stylesheet_data["rules"].append({"handler_type": "codergen", "model": model})
    stylesheet = ModelStylesheet.from_dict(stylesheet_data)

    interviewer = CLIInterviewer(console=console)
    registry = create_default_registry(
        pipeline=pipeline, interviewer=interviewer
    )

    engine = PipelineEngine(
        registry=registry,
        stylesheet=stylesheet,
        checkpoint_dir=checkpoint_dir,
    )

    console.print(f"[bold green]Running pipeline:[/bold green] {pipeline.name}")
    ctx = asyncio.run(engine.run(pipeline))

    console.print("[bold green]Pipeline completed.[/bold green]")
    _print_context(ctx)


@main.command()
@click.argument("pipeline_dot", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Treat warnings as errors.")
def validate(pipeline_dot: str, strict: bool) -> None:
    """Validate a pipeline DOT file without executing it."""
    try:
        pipeline = parse_dot_file(pipeline_dot)
    except Exception as exc:
        console.print(f"[red]Failed to parse pipeline:[/red] {exc}")
        raise SystemExit(1) from exc

    findings = validate_pipeline(pipeline)

    if not findings:
        console.print("[green]Pipeline is valid.[/green]")
        return

    table = Table(title="Validation Results")
    table.add_column("Level", style="bold")
    table.add_column("Location")
    table.add_column("Message")

    for f in findings:
        level_style = "red" if f.level == ValidationLevel.ERROR else "yellow"
        location = f.node_name or ""
        if f.edge:
            location = f"{f.edge.source} -> {f.edge.target}"
        table.add_row(f"[{level_style}]{f.level.value}[/{level_style}]", location, f.message)

    console.print(table)

    if has_errors(findings) or (strict and findings):
        raise SystemExit(1)


@main.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--pipeline-dot",
    type=click.Path(exists=True),
    default=None,
    help="Pipeline DOT file (uses checkpoint metadata if omitted).",
)
def resume(checkpoint_path: str, verbose: bool, pipeline_dot: str | None) -> None:
    """Resume a pipeline from a checkpoint file."""
    _setup_logging(verbose)

    try:
        cp = Checkpoint.load_from_file(checkpoint_path)
    except Exception as exc:
        console.print(f"[red]Failed to load checkpoint:[/red] {exc}")
        raise SystemExit(1) from exc

    if pipeline_dot is None:
        console.print("[red]--pipeline-dot is required for resume[/red]")
        raise SystemExit(1)

    try:
        pipeline = parse_dot_file(pipeline_dot)
    except Exception as exc:
        console.print(f"[red]Failed to parse pipeline:[/red] {exc}")
        raise SystemExit(1) from exc

    interviewer = CLIInterviewer(console=console)
    registry = create_default_registry(
        pipeline=pipeline, interviewer=interviewer
    )

    engine = PipelineEngine(registry=registry)

    console.print(
        f"[bold green]Resuming pipeline:[/bold green] {cp.pipeline_name} "
        f"from node '{cp.current_node}'"
    )
    ctx = asyncio.run(engine.run(pipeline, checkpoint=cp))

    console.print("[bold green]Pipeline completed.[/bold green]")
    _print_context(ctx)


def _print_context(ctx: PipelineContext) -> None:
    """Print the final pipeline context in a table."""
    data = ctx.to_dict()
    if not data:
        return

    table = Table(title="Final Context")
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    for key, value in sorted(data.items()):
        if key.startswith("_"):
            continue  # skip internal keys
        table.add_row(key, str(value)[:200])

    console.print(table)


if __name__ == "__main__":
    main()
