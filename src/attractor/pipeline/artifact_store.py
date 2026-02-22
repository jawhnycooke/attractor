"""Artifact store for large pipeline stage outputs.

Provides named, typed storage for artifacts that are too large for the
context blackboard.  Small artifacts live in memory; those exceeding
a configurable threshold are persisted to disk.

See spec Section 5.5.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default file-backing threshold per spec: 100 KB
FILE_BACKING_THRESHOLD = 100 * 1024


@dataclass
class ArtifactInfo:
    """Metadata for a stored artifact.

    Attributes:
        id: Unique artifact identifier.
        name: Human-readable name.
        size_bytes: Size of the artifact data in bytes.
        stored_at: UNIX timestamp when the artifact was stored.
        is_file_backed: Whether the data is persisted on disk.
    """

    id: str
    name: str
    size_bytes: int
    stored_at: float = field(default_factory=time.time)
    is_file_backed: bool = False


@runtime_checkable
class ArtifactStore(Protocol):
    """Protocol for artifact storage backends."""

    def store(self, artifact_id: str, name: str, data: str | bytes) -> ArtifactInfo:
        """Store an artifact and return its metadata.

        Args:
            artifact_id: Unique identifier for the artifact.
            name: Human-readable name.
            data: The artifact payload (str or bytes).

        Returns:
            Metadata about the stored artifact.
        """
        ...

    def retrieve(self, artifact_id: str) -> str | bytes:
        """Retrieve artifact data by ID.

        Args:
            artifact_id: The artifact to retrieve.

        Returns:
            The stored data.

        Raises:
            KeyError: If the artifact does not exist.
        """
        ...

    def has(self, artifact_id: str) -> bool:
        """Check whether an artifact exists.

        Args:
            artifact_id: The artifact to check.

        Returns:
            True if the artifact exists.
        """
        ...

    def list(self) -> list[ArtifactInfo]:
        """Return metadata for all stored artifacts.

        Returns:
            List of ArtifactInfo for every artifact in the store.
        """
        ...

    def remove(self, artifact_id: str) -> None:
        """Remove an artifact by ID.

        Args:
            artifact_id: The artifact to remove.

        Raises:
            KeyError: If the artifact does not exist.
        """
        ...

    def clear(self) -> None:
        """Remove all artifacts from the store."""
        ...


def _byte_size(data: str | bytes) -> int:
    """Return the size of *data* in bytes."""
    if isinstance(data, bytes):
        return len(data)
    return len(data.encode("utf-8"))


class LocalArtifactStore:
    """In-memory artifact store with optional file-backing for large items.

    Artifacts smaller than *threshold* bytes are kept in a dict.
    Larger artifacts are written to ``{base_dir}/artifacts/`` as JSON files.

    Args:
        base_dir: Directory for file-backed artifacts.  If ``None``,
            all artifacts are kept in memory regardless of size.
        threshold: Size in bytes above which artifacts are file-backed.
            Defaults to 100 KB.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        threshold: int = FILE_BACKING_THRESHOLD,
    ) -> None:
        self._base_dir: Path | None = Path(base_dir) if base_dir else None
        self._threshold = threshold
        self._artifacts: dict[str, tuple[ArtifactInfo, str | bytes | Path]] = {}
        self._lock = threading.Lock()

    def store(self, artifact_id: str, name: str, data: str | bytes) -> ArtifactInfo:
        """Store an artifact, file-backing if above threshold."""
        size = _byte_size(data)
        is_file_backed = size > self._threshold and self._base_dir is not None

        if is_file_backed:
            assert self._base_dir is not None
            artifacts_dir = self._base_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            file_path = artifacts_dir / f"{artifact_id}.json"
            if isinstance(data, bytes):
                file_path.write_bytes(data)
            else:
                file_path.write_text(data, encoding="utf-8")
            stored: str | bytes | Path = file_path
            logger.debug(
                "Artifact '%s' file-backed to %s (%d bytes)",
                artifact_id,
                file_path,
                size,
            )
        else:
            stored = data
            logger.debug(
                "Artifact '%s' stored in memory (%d bytes)", artifact_id, size
            )

        info = ArtifactInfo(
            id=artifact_id,
            name=name,
            size_bytes=size,
            is_file_backed=is_file_backed,
        )

        with self._lock:
            self._artifacts[artifact_id] = (info, stored)

        return info

    def retrieve(self, artifact_id: str) -> str | bytes:
        """Retrieve artifact data by ID."""
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact not found: {artifact_id!r}")
            info, stored = self._artifacts[artifact_id]

        if info.is_file_backed:
            assert isinstance(stored, Path)
            # Return as the same type originally stored
            try:
                return stored.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return stored.read_bytes()

        assert isinstance(stored, (str, bytes))
        return stored

    def has(self, artifact_id: str) -> bool:
        """Check whether an artifact exists."""
        with self._lock:
            return artifact_id in self._artifacts

    def list(self) -> list[ArtifactInfo]:
        """Return metadata for all stored artifacts."""
        with self._lock:
            return [info for info, _ in self._artifacts.values()]

    def remove(self, artifact_id: str) -> None:
        """Remove an artifact by ID."""
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"Artifact not found: {artifact_id!r}")
            info, stored = self._artifacts.pop(artifact_id)

        # Clean up file if file-backed
        if info.is_file_backed and isinstance(stored, Path) and stored.exists():
            stored.unlink()
            logger.debug("Removed file-backed artifact '%s'", artifact_id)

    def clear(self) -> None:
        """Remove all artifacts from the store."""
        with self._lock:
            items = list(self._artifacts.items())
            self._artifacts.clear()

        # Clean up files
        for _, (info, stored) in items:
            if info.is_file_backed and isinstance(stored, Path) and stored.exists():
                stored.unlink()
