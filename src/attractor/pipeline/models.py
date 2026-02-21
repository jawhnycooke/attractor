"""Pipeline data models.

Defines the core structures for pipeline definition, execution state,
and node results used throughout the pipeline engine.
"""

from __future__ import annotations

import copy
import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class OutcomeStatus(str, enum.Enum):
    """Handler outcome status per spec."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RETRY = "retry"
    FAIL = "fail"
    SKIPPED = "skipped"


_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(value: str) -> float:
    """Parse a duration string like '900s', '15m', '250ms' into seconds."""
    value = value.strip()
    for suffix, multiplier in sorted(
        _DURATION_UNITS.items(), key=lambda x: -len(x[0])
    ):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier
    raise ValueError(f"Invalid duration: {value!r}")


def coerce_value(raw: str) -> str | int | float | bool:
    """Coerce a raw string attribute value to its typed form."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


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
        label: Human-readable label for visualization.
        shape: GraphViz node shape.
        classes: CSS-like class list for stylesheet matching.
        prompt: Prompt template for LLM-backed handlers.
        max_retries: Maximum retry count for this node.
        goal_gate: Whether this node is a goal gate.
        retry_target: Node to jump to on retry.
        fallback_retry_target: Fallback retry target if retry_target is unset.
        fidelity: Fidelity level override for this node.
        thread_id: Thread grouping identifier.
        timeout: Maximum execution time in seconds.
        llm_model: LLM model override for this node.
        llm_provider: LLM provider override for this node.
        reasoning_effort: Reasoning effort level for LLM calls.
        auto_status: Whether to auto-report status updates.
        allow_partial: Whether partial success is acceptable.
    """

    name: str
    handler_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    is_start: bool = False
    is_terminal: bool = False
    label: str = ""
    shape: str = "box"
    classes: list[str] = field(default_factory=list)
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str | None = None
    fallback_retry_target: str | None = None
    fidelity: str | None = None
    thread_id: str | None = None
    timeout: float | None = None
    llm_model: str | None = None
    llm_provider: str | None = None
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False


@dataclass
class PipelineEdge:
    """A directed edge between two pipeline nodes.

    Attributes:
        source: Name of the originating node.
        target: Name of the destination node.
        condition: Optional expression evaluated against PipelineContext.
            ``None`` means unconditional (default edge).
        label: Human-readable label for visualization.
        weight: Ordering hint when multiple edges share a source.
            Higher values are evaluated first (descending sort).
        fidelity: Fidelity level override for this edge transition.
        thread_id: Thread grouping identifier for this edge.
        loop_restart: Whether traversing this edge restarts a loop.
    """

    source: str
    target: str
    condition: str | None = None
    label: str = ""
    weight: int = 0
    fidelity: str | None = None
    thread_id: str | None = None
    loop_restart: bool = False


@dataclass
class Pipeline:
    """Complete pipeline definition parsed from a DOT file.

    Attributes:
        name: Pipeline identifier (typically the DOT graph name).
        nodes: Mapping of node name -> PipelineNode.
        edges: All directed edges in the graph.
        start_node: Name of the designated start node.
        metadata: Arbitrary metadata from the graph-level attributes.
        goal: Top-level goal description for the pipeline.
        default_max_retry: Default max retries for nodes without explicit setting.
        retry_target: Default retry target node.
        fallback_retry_target: Fallback retry target if retry_target is unset.
        default_fidelity: Default fidelity level for all nodes.
        model_stylesheet: Path to model stylesheet file.
    """

    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[PipelineEdge] = field(default_factory=list)
    start_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    goal: str = ""
    label: str = ""
    default_max_retry: int = 50
    retry_target: str | None = None
    fallback_retry_target: str | None = None
    default_fidelity: str | None = None
    model_stylesheet: str | None = None

    def outgoing_edges(self, node_name: str) -> list[PipelineEdge]:
        """Return edges originating from *node_name*, sorted by weight desc, then target asc."""
        return sorted(
            [e for e in self.edges if e.source == node_name],
            key=lambda e: (-e.weight, e.target),
        )

    def incoming_edges(self, node_name: str) -> list[PipelineEdge]:
        """Return edges targeting *node_name*."""
        return [e for e in self.edges if e.target == node_name]


@dataclass
class NodeResult:
    """Result returned by a node handler after execution.

    Attributes:
        status: Outcome status of the handler execution.
        output: Arbitrary output payload.
        failure_reason: Error description when status indicates failure.
        next_node: Explicit routing override — bypasses edge evaluation.
        context_updates: Key-value pairs to merge into PipelineContext.
        preferred_label: Preferred edge label for routing.
        suggested_next_ids: Suggested next node IDs for routing hints.
        notes: Free-form notes from the handler.
    """

    status: OutcomeStatus
    output: Any = None
    failure_reason: str | None = None
    next_node: str | None = None
    context_updates: dict[str, Any] = field(default_factory=dict)
    preferred_label: str | None = None
    suggested_next_ids: list[str] = field(default_factory=list)
    notes: str | None = None

    @property
    def success(self) -> bool:
        """Backward-compatible success check."""
        return self.status in (OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL_SUCCESS)


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

    def clone(self) -> PipelineContext:
        """Create a deep copy of this context for parallel branch isolation.

        Returns:
            A new :class:`PipelineContext` with deep-copied data.
        """
        ctx = PipelineContext()
        ctx._data = copy.deepcopy(self._data)
        return ctx

    def create_scope(self, prefix: str) -> PipelineContext:
        """Create a child context whose keys are prefixed on merge-back.

        The child starts as a deep copy of the parent so that parallel
        branches see the same initial state but cannot mutate each other.

        Args:
            prefix: The prefix applied when merging back.

        Returns:
            A new :class:`PipelineContext` cloned from this one.
        """
        return self.clone()

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
        node_retries: Per-node retry counts at checkpoint time.
        logs: Structured log entries captured during execution.
    """

    pipeline_name: str
    current_node: str
    context: PipelineContext
    completed_nodes: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    node_retries: dict[str, int] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

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
            "node_retries": dict(self.node_retries),
            "logs": list(self.logs),
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
            node_retries=dict(data.get("node_retries", {})),
            logs=list(data.get("logs", [])),
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
