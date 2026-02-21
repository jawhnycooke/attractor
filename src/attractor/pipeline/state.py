"""Pipeline state management.

Re-exports :class:`PipelineContext` and :class:`Checkpoint` from models
and adds file-system helpers for checkpoint persistence.
"""

from __future__ import annotations

from pathlib import Path

from attractor.pipeline.models import Checkpoint, PipelineContext

__all__ = ["PipelineContext", "Checkpoint", "list_checkpoints", "latest_checkpoint"]


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Return all checkpoint JSON files in *directory*, newest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("checkpoint_*.json"), reverse=True)
    return files


def latest_checkpoint(directory: str | Path) -> Checkpoint | None:
    """Load the most recent checkpoint from *directory*, or ``None``."""
    files = list_checkpoints(directory)
    if not files:
        return None
    return Checkpoint.load_from_file(files[0])


def save_checkpoint(checkpoint: Checkpoint, directory: str | Path) -> Path:
    """Persist *checkpoint* to *directory* with a timestamp-based filename.

    Returns:
        Path to the written checkpoint file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_{int(checkpoint.timestamp * 1000)}.json"
    path = directory / filename
    checkpoint.save_to_file(path)
    return path
