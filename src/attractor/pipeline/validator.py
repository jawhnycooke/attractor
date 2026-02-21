"""Pipeline validation.

Statically checks a :class:`Pipeline` for structural errors and
warnings before execution.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from attractor.pipeline.conditions import validate_condition_syntax
from attractor.pipeline.models import Pipeline, PipelineEdge

# Valid fidelity mode values per spec §5.4
_VALID_FIDELITY_MODES = {
    "full",
    "truncate",
    "compact",
    "summary:low",
    "summary:medium",
    "summary:high",
}

# Handler types that are known to the built-in registry
_KNOWN_HANDLERS = {
    "start",
    "exit",
    "codergen",
    "wait.human",
    "conditional",
    "parallel",
    "parallel.fan_in",
    "tool",
    "stack.manager_loop",
}

# Required attributes per handler type
_REQUIRED_ATTRS: dict[str, list[str]] = {
    "tool": ["tool_command"],
}


class ValidationLevel(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """A single validation finding."""

    level: ValidationLevel
    message: str
    node_name: str | None = None
    edge: PipelineEdge | None = None
    rule: str = ""
    fix: str | None = None

    def __str__(self) -> str:
        location = ""
        if self.node_name:
            location = f" (node '{self.node_name}')"
        elif self.edge:
            location = f" (edge {self.edge.source} -> {self.edge.target})"
        rule_tag = f" [{self.rule}]" if self.rule else ""
        return f"[{self.level.value.upper()}]{location}{rule_tag} {self.message}"


@runtime_checkable
class LintRule(Protocol):
    """Protocol for custom validation rules."""

    name: str

    def check(self, pipeline: Pipeline) -> list[ValidationError]: ...


_custom_rules: list[LintRule] = []


def register_lint_rule(rule: LintRule) -> None:
    """Register a custom lint rule to run during validation."""
    _custom_rules.append(rule)


def validate_pipeline(
    pipeline: Pipeline,
    extra_rules: list[LintRule] | None = None,
) -> list[ValidationError]:
    """Run all validation checks on *pipeline*.

    Args:
        pipeline: The pipeline to validate.
        extra_rules: Additional lint rules to run.

    Returns:
        A list of :class:`ValidationError` findings, possibly empty.
    """
    errors: list[ValidationError] = []

    _check_start_node(pipeline, errors)
    _check_start_no_incoming(pipeline, errors)
    _check_terminal_nodes(pipeline, errors)
    _check_exit_no_outgoing(pipeline, errors)
    _check_handler_types(pipeline, errors)
    _check_required_attributes(pipeline, errors)
    _check_edge_references(pipeline, errors)
    _check_condition_syntax(pipeline, errors)
    _check_reachability(pipeline, errors)
    _check_cycles(pipeline, errors)
    _check_retry_target_exists(pipeline, errors)
    _check_goal_gate_has_retry(pipeline, errors)
    _check_prompt_on_llm_nodes(pipeline, errors)
    _check_stylesheet_syntax(pipeline, errors)
    _check_fidelity_valid(pipeline, errors)

    for rule in _custom_rules:
        errors.extend(rule.check(pipeline))

    if extra_rules:
        for rule in extra_rules:
            errors.extend(rule.check(pipeline))

    return errors


def has_errors(findings: list[ValidationError]) -> bool:
    """Return True if any finding is an error (not just a warning)."""
    return any(f.level == ValidationLevel.ERROR for f in findings)


class ValidationException(Exception):
    """Raised by :func:`validate_or_raise` when ERROR-level diagnostics exist."""

    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        messages = [str(e) for e in errors]
        super().__init__("\n".join(messages))


def validate_or_raise(
    pipeline: Pipeline,
    extra_rules: list[LintRule] | None = None,
) -> list[ValidationError]:
    """Run validation and raise on any ERROR-level diagnostic.

    Args:
        pipeline: The pipeline to validate.
        extra_rules: Additional lint rules to run.

    Returns:
        The full list of diagnostics (only warnings/info if no exception).

    Raises:
        ValidationException: If any ERROR-level diagnostics are found.
    """
    findings = validate_pipeline(pipeline, extra_rules=extra_rules)
    error_findings = [f for f in findings if f.level == ValidationLevel.ERROR]
    if error_findings:
        raise ValidationException(error_findings)
    return findings


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_start_node(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    if not pipeline.start_node:
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message="No start node defined",
                rule="start_node",
            )
        )
    elif pipeline.start_node not in pipeline.nodes:
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Start node '{pipeline.start_node}' does not exist",
                rule="start_node",
            )
        )
    else:
        # Exactly one start node required
        start_nodes = [n for n in pipeline.nodes.values() if n.is_start]
        if len(start_nodes) > 1:
            names = ", ".join(n.name for n in start_nodes)
            errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Multiple start nodes found: {names}",
                    rule="start_node",
                    fix="Ensure exactly one node has is_start=true or shape=Mdiamond",
                )
            )


def _check_start_no_incoming(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    if not pipeline.start_node:
        return
    incoming = pipeline.incoming_edges(pipeline.start_node)
    if incoming:
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message="Start node must not have incoming edges",
                node_name=pipeline.start_node,
                rule="start_no_incoming",
            )
        )


def _check_terminal_nodes(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    terminals = [n for n in pipeline.nodes.values() if n.is_terminal]
    if not terminals:
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message="No terminal nodes found",
                rule="terminal_node",
            )
        )
    elif len(terminals) > 1:
        names = ", ".join(n.name for n in terminals)
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Multiple terminal nodes found: {names}",
                rule="terminal_node",
                fix="Ensure exactly one node is terminal (shape=Msquare or named 'exit'/'end')",
            )
        )


def _check_exit_no_outgoing(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        if node.handler_type == "exit":
            outgoing = pipeline.outgoing_edges(node.name)
            if outgoing:
                errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        message="Exit node must not have outgoing edges",
                        node_name=node.name,
                        rule="exit_no_outgoing",
                    )
                )


def _check_handler_types(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    for node in pipeline.nodes.values():
        if node.handler_type not in _KNOWN_HANDLERS:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message=f"Unknown handler type '{node.handler_type}'",
                    node_name=node.name,
                    rule="type_known",
                )
            )


def _check_required_attributes(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        required = _REQUIRED_ATTRS.get(node.handler_type, [])
        for attr in required:
            if attr not in node.attributes:
                errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        message=(
                            f"Missing required attribute '{attr}' "
                            f"for handler '{node.handler_type}'"
                        ),
                        node_name=node.name,
                    )
                )


def _check_edge_references(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    for edge in pipeline.edges:
        if edge.source not in pipeline.nodes:
            errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Edge source '{edge.source}' does not exist",
                    edge=edge,
                    rule="edge_target_exists",
                )
            )
        if edge.target not in pipeline.nodes:
            errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Edge target '{edge.target}' does not exist",
                    edge=edge,
                    rule="edge_target_exists",
                )
            )


def _check_condition_syntax(pipeline: Pipeline, errors: list[ValidationError]) -> None:
    for edge in pipeline.edges:
        if edge.condition:
            err = validate_condition_syntax(edge.condition)
            if err:
                errors.append(
                    ValidationError(
                        level=ValidationLevel.ERROR,
                        message=f"Invalid condition syntax: {err}",
                        edge=edge,
                        rule="condition_syntax",
                    )
                )


def _check_reachability(pipeline: Pipeline, errors: list[ValidationError]) -> None:
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
            errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    message="Node is unreachable from start",
                    node_name=name,
                    rule="reachability",
                )
            )


def _check_cycles(pipeline: Pipeline, errors: list[ValidationError]) -> None:
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
                errors.append(
                    ValidationError(
                        level=ValidationLevel.WARNING,
                        message="Node is part of a non-supervisor cycle",
                        node_name=n,
                    )
                )


def _check_retry_target_exists(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        if node.retry_target and node.retry_target not in pipeline.nodes:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message=f"retry_target '{node.retry_target}' does not exist",
                    node_name=node.name,
                    rule="retry_target_exists",
                )
            )
        if (
            node.fallback_retry_target
            and node.fallback_retry_target not in pipeline.nodes
        ):
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message=(
                        f"fallback_retry_target '{node.fallback_retry_target}' "
                        "does not exist"
                    ),
                    node_name=node.name,
                    rule="retry_target_exists",
                )
            )


def _check_goal_gate_has_retry(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        if node.goal_gate and not node.retry_target and not pipeline.retry_target:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message="goal_gate node has no retry_target",
                    node_name=node.name,
                    rule="goal_gate_has_retry",
                    fix="Add retry_target attribute or set graph-level retry_target",
                )
            )


def _check_prompt_on_llm_nodes(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    for node in pipeline.nodes.values():
        if node.handler_type == "codergen" and not node.prompt and not node.label:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message="LLM node has no prompt or label",
                    node_name=node.name,
                    rule="prompt_on_llm_nodes",
                    fix="Add a prompt or label attribute",
                )
            )


def _check_stylesheet_syntax(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    """Check that model_stylesheet parses as valid stylesheet rules (V1)."""
    if not pipeline.model_stylesheet:
        return

    try:
        from attractor.pipeline.stylesheet import parse_stylesheet

        result = parse_stylesheet(pipeline.model_stylesheet)
        if not result.rules:
            errors.append(
                ValidationError(
                    level=ValidationLevel.ERROR,
                    message="model_stylesheet contains no valid rules",
                    rule="stylesheet_syntax",
                    fix="Ensure stylesheet uses valid selectors (*, #id, .class) and declarations",
                )
            )
    except Exception as exc:
        errors.append(
            ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Invalid model_stylesheet syntax: {exc}",
                rule="stylesheet_syntax",
            )
        )


def _check_fidelity_valid(
    pipeline: Pipeline, errors: list[ValidationError]
) -> None:
    """Check that fidelity values are valid (V2)."""
    # Check graph-level default_fidelity
    if pipeline.default_fidelity and pipeline.default_fidelity not in _VALID_FIDELITY_MODES:
        errors.append(
            ValidationError(
                level=ValidationLevel.WARNING,
                message=(
                    f"Invalid default_fidelity '{pipeline.default_fidelity}'; "
                    f"must be one of: {', '.join(sorted(_VALID_FIDELITY_MODES))}"
                ),
                rule="fidelity_valid",
            )
        )

    # Check per-node fidelity
    for node in pipeline.nodes.values():
        if node.fidelity and node.fidelity not in _VALID_FIDELITY_MODES:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message=(
                        f"Invalid fidelity '{node.fidelity}'; "
                        f"must be one of: {', '.join(sorted(_VALID_FIDELITY_MODES))}"
                    ),
                    node_name=node.name,
                    rule="fidelity_valid",
                )
            )

    # Check per-edge fidelity
    for edge in pipeline.edges:
        if edge.fidelity and edge.fidelity not in _VALID_FIDELITY_MODES:
            errors.append(
                ValidationError(
                    level=ValidationLevel.WARNING,
                    message=(
                        f"Invalid fidelity '{edge.fidelity}'; "
                        f"must be one of: {', '.join(sorted(_VALID_FIDELITY_MODES))}"
                    ),
                    edge=edge,
                    rule="fidelity_valid",
                )
            )
