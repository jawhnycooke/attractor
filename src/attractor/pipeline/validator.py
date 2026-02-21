"""Pipeline validation.

Statically checks a :class:`Pipeline` for structural errors and
warnings before execution.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from attractor.pipeline.conditions import validate_condition_syntax
from attractor.pipeline.models import Pipeline, PipelineEdge

# Handler types that are known to the built-in registry
_KNOWN_HANDLERS = {
    "codergen",
    "human_gate",
    "conditional",
    "parallel",
    "tool",
    "supervisor",
}

# Required attributes per handler type
_REQUIRED_ATTRS: dict[str, list[str]] = {
    "tool": ["command"],
}


class ValidationLevel(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationError:
    """A single validation finding."""

    level: ValidationLevel
    message: str
    node_name: str | None = None
    edge: PipelineEdge | None = None

    def __str__(self) -> str:
        location = ""
        if self.node_name:
            location = f" (node '{self.node_name}')"
        elif self.edge:
            location = f" (edge {self.edge.source} -> {self.edge.target})"
        return f"[{self.level.value.upper()}]{location} {self.message}"


def validate_pipeline(pipeline: Pipeline) -> list[ValidationError]:
    """Run all validation checks on *pipeline*.

    Returns:
        A list of :class:`ValidationError` findings, possibly empty.
    """
    errors: list[ValidationError] = []

    _check_start_node(pipeline, errors)
    _check_terminal_nodes(pipeline, errors)
    _check_handler_types(pipeline, errors)
    _check_required_attributes(pipeline, errors)
    _check_edge_references(pipeline, errors)
    _check_condition_syntax(pipeline, errors)
    _check_reachability(pipeline, errors)
    _check_cycles(pipeline, errors)

    return errors


def has_errors(findings: list[ValidationError]) -> bool:
    """Return True if any finding is an error (not just a warning)."""
    return any(f.level == ValidationLevel.ERROR for f in findings)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_start_node(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    if not pipeline.start_node:
        errors.append(ValidationError(
            level=ValidationLevel.ERROR,
            message="No start node defined",
        ))
    elif pipeline.start_node not in pipeline.nodes:
        errors.append(ValidationError(
            level=ValidationLevel.ERROR,
            message=f"Start node '{pipeline.start_node}' does not exist",
        ))


def _check_terminal_nodes(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    terminals = [n for n in pipeline.nodes.values() if n.is_terminal]
    if not terminals:
        errors.append(ValidationError(
            level=ValidationLevel.ERROR,
            message="No terminal nodes found",
        ))


def _check_handler_types(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        if node.handler_type not in _KNOWN_HANDLERS:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Unknown handler type '{node.handler_type}'",
                node_name=node.name,
            ))


def _check_required_attributes(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        required = _REQUIRED_ATTRS.get(node.handler_type, [])
        for attr in required:
            if attr not in node.attributes:
                errors.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Missing required attribute '{attr}' for handler '{node.handler_type}'",
                    node_name=node.name,
                ))


def _check_edge_references(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for edge in pipeline.edges:
        if edge.source not in pipeline.nodes:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Edge source '{edge.source}' does not exist",
                edge=edge,
            ))
        if edge.target not in pipeline.nodes:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Edge target '{edge.target}' does not exist",
                edge=edge,
            ))


def _check_condition_syntax(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for edge in pipeline.edges:
        if edge.condition:
            err = validate_condition_syntax(edge.condition)
            if err:
                errors.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Invalid condition syntax: {err}",
                    edge=edge,
                ))


def _check_reachability(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    """Warn about nodes unreachable from the start node."""
    if not pipeline.start_node or pipeline.start_node not in pipeline.nodes:
        return  # already covered by _check_start_node

    reachable: set[str] = set()
    queue: deque[str] = deque([pipeline.start_node])
    adj: dict[str, list[str]] = {}
    for edge in pipeline.edges:
        adj.setdefault(edge.source, []).append(edge.target)

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for neighbor in adj.get(current, []):
            if neighbor not in reachable:
                queue.append(neighbor)

    for name in pipeline.nodes:
        if name not in reachable:
            errors.append(ValidationError(
                level=ValidationLevel.WARNING,
                message=f"Node is unreachable from start",
                node_name=name,
            ))


def _check_cycles(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    """Warn about cycles not involving a supervisor handler.

    Cycles through supervisor nodes are expected (iterative refinement).
    Other cycles may indicate configuration errors.
    """
    adj: dict[str, list[str]] = {}
    for edge in pipeline.edges:
        adj.setdefault(edge.source, []).append(edge.target)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in pipeline.nodes}
    cycle_nodes: list[str] = []

    def dfs(node: str, path: list[str]) -> bool:
        color[node] = GRAY
        path.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                # Found a cycle — check if any node in the cycle is a supervisor
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                has_supervisor = any(
                    pipeline.nodes[n].handler_type == "supervisor"
                    for n in cycle
                    if n in pipeline.nodes
                )
                if not has_supervisor:
                    cycle_nodes.extend(cycle)
                return False  # continue DFS (don't abort)
            if color[neighbor] == WHITE:
                dfs(neighbor, path)
        path.pop()
        color[node] = BLACK
        return False

    for node in pipeline.nodes:
        if color.get(node) == WHITE:
            dfs(node, [])

    if cycle_nodes:
        seen: set[str] = set()
        for n in cycle_nodes:
            if n not in seen:
                seen.add(n)
                errors.append(ValidationError(
                    level=ValidationLevel.WARNING,
                    message="Node is part of a non-supervisor cycle",
                    node_name=n,
                ))
