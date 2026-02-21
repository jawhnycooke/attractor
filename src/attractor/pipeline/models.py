"""Pipeline data models.

Defines the core structures for pipeline definition, execution state,
and node results used throughout the pipeline engine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineNode:
    """A single node in the pipeline graph.

    Attributes:
        name: Unique identifier for this node.
        handler_type: Dispatch key for the handler registry
            (e.g. "codergen", "human_gate", "tool").
        attributes: Handler-specific configuration extracted from DOT
            attributes or stylesheet overrides.
        is_start: Whether this is the entry point of the pipeline.
        is_terminal: Whether this is a terminal (exit) node.
    """

    name: str
    handler_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    is_start: bool = False
    is_terminal: bool = False


@dataclass
class PipelineEdge:
    """A directed edge between two pipeline nodes.

    Attributes:
        source: Name of the originating node.
        target: Name of the destination node.
        condition: Optional expression evaluated against PipelineContext.
            ``None`` means unconditional (default edge).
        label: Human-readable label for visualization.
        priority: Ordering hint when multiple edges share a source.
            Lower values are evaluated first.
    """

    source: str
    target: str
    condition: str | None = None
    label: str = ""
    priority: int = 0


@dataclass
class Pipeline:
    """Complete pipeline definition parsed from a DOT file.

    Attributes:
        name: Pipeline identifier (typically the DOT graph name).
        nodes: Mapping of node name -> PipelineNode.
        edges: All directed edges in the graph.
        start_node: Name of the designated start node.
        metadata: Arbitrary metadata from the graph-level attributes.
    """

    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[PipelineEdge] = field(default_factory=list)
    start_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def outgoing_edges(self, node_name: str) -> list[PipelineEdge]:
        """Return edges originating from *node_name*, sorted by priority."""
        return sorted(
            [e for e in self.edges if e.source == node_name],
            key=lambda e: e.priority,
        )

    def incoming_edges(self, node_name: str) -> list[PipelineEdge]:
        """Return edges targeting *node_name*."""
        return [e for e in self.edges if e.target == node_name]


@dataclass
class NodeResult:
    """Result returned by a node handler after execution.

    Attributes:
        success: Whether the handler completed without error.
        output: Arbitrary output payload.
        error: Error description when ``success`` is False.
        next_node: Explicit routing override — bypasses edge evaluation.
        context_updates: Key-value pairs to merge into PipelineContext.
    """

    success: bool
    output: Any = None
    error: str | None = None
    next_node: str | None = None
    context_updates: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Shared key-value state store for pipeline execution.

    Acts as the "blackboard" that nodes read from and write to.
    Supports scoped sub-contexts for parallel branch isolation and
    JSON serialization for checkpointing.

    Data is stored privately in ``_data``.  Use the accessor methods
    (:meth:`get`, :meth:`set`, :meth:`has`, :meth:`delete`,
    :meth:`to_dict`, :meth:`update`) to interact with the store.
    """

    _data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent.

        Args:
            key: The context key to look up.
            default: Value returned when *key* is not present.

        Returns:
            The stored value or *default*.
        """
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*.

        Args:
            key: The context key.
            value: The value to store.
        """
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* exists in the context.

        Args:
            key: The context key to check.

        Returns:
            Whether the key is present.
        """
        return key in self._data

    def delete(self, key: str) -> None:
        """Remove *key* from the context if present.

        Args:
            key: The context key to remove.
        """
        self._data.pop(key, None)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of all context data.

        Returns:
            A new ``dict`` with all key-value pairs.
        """
        return dict(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineContext:
        """Create a context from an existing dictionary.

        Args:
            data: Initial key-value pairs.

        Returns:
            A new :class:`PipelineContext` populated with *data*.
        """
        ctx = cls()
        ctx._data = dict(data)
        return ctx

    def create_scope(self, prefix: str) -> PipelineContext:
        """Create a child context whose keys are prefixed on merge-back.

        Args:
            prefix: The prefix applied when merging back.

        Returns:
            A new empty :class:`PipelineContext`.
        """
        return PipelineContext()

    def merge_scope(self, scope: PipelineContext, prefix: str) -> None:
        """Merge a scoped context back, prefixing its keys.

        Args:
            scope: The child context to merge.
            prefix: Prefix prepended to each key.
        """
        for key, value in scope._data.items():
            self._data[f"{prefix}.{key}"] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into the context.

        Args:
            updates: Key-value pairs to merge.
        """
        self._data.update(updates)


@dataclass
class Checkpoint:
    """Serializable execution snapshot for resume-on-failure.

    Attributes:
        pipeline_name: Name of the pipeline being executed.
        current_node: Node that was about to execute (or just completed).
        context: Full pipeline context at checkpoint time.
        completed_nodes: Ordered list of nodes that finished successfully.
        timestamp: UNIX epoch when the checkpoint was created.
    """

    pipeline_name: str
    current_node: str
    context: PipelineContext
    completed_nodes: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint to a plain dictionary.

        Returns:
            A JSON-compatible ``dict`` representing this checkpoint.
        """
        return {
            "pipeline_name": self.pipeline_name,
            "current_node": self.current_node,
            "context": self.context.to_dict(),
            "completed_nodes": list(self.completed_nodes),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Reconstruct a checkpoint from a dictionary.

        Args:
            data: Dictionary with checkpoint fields (as produced by
                :meth:`to_dict`).

        Returns:
            A new :class:`Checkpoint` instance.
        """
        return cls(
            pipeline_name=data["pipeline_name"],
            current_node=data["current_node"],
            context=PipelineContext.from_dict(data["context"]),
            completed_nodes=list(data["completed_nodes"]),
            timestamp=data["timestamp"],
        )

    def save_to_file(self, path: str | Path) -> None:
        """Write this checkpoint to a JSON file.

        Args:
            path: Destination file path.  Parent directories are
                created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_from_file(cls, path: str | Path) -> Checkpoint:
        """Load a checkpoint from a JSON file.

        Args:
            path: Path to the checkpoint file.

        Returns:
            A new :class:`Checkpoint` instance.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))
