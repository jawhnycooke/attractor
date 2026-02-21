"""DOT file parser for pipeline definitions.

Uses the ``pydot`` library to parse GraphViz DOT files into
:class:`~attractor.pipeline.models.Pipeline` objects.  Node and edge
attributes are extracted and mapped onto the pipeline data model.

Handler resolution is shape-based: a node's GraphViz shape selects its
handler type via :data:`SHAPE_HANDLER_MAP`.  An explicit ``type``
attribute overrides the shape-derived handler.  The ``Mdiamond`` shape
marks the start node and ``Msquare`` marks exit (terminal) nodes.
Nodes with no outgoing edges are implicitly terminal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pydot

from attractor.pipeline.models import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
    coerce_value,
    parse_duration,
)


class ParseError(Exception):
    """Raised when a DOT file cannot be parsed into a valid pipeline."""


SHAPE_HANDLER_MAP: dict[str, str] = {
    "Mdiamond": "start",
    "Msquare": "exit",
    "box": "codergen",
    "hexagon": "wait.human",
    "diamond": "conditional",
    "component": "parallel",
    "tripleoctagon": "parallel.fan_in",
    "parallelogram": "tool",
    "house": "stack.manager_loop",
}


def parse_dot_file(path: str | Path) -> Pipeline:
    """Parse a DOT file at *path* and return a :class:`Pipeline`.

    Raises:
        ParseError: If the file cannot be read or parsed.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    return parse_dot_string(path.read_text(), name=path.stem)


def parse_dot_string(dot_content: str, name: str = "pipeline") -> Pipeline:
    """Parse a DOT string and return a :class:`Pipeline`.

    Args:
        dot_content: Raw DOT source text.
        name: Fallback pipeline name if the graph has no name.

    Raises:
        ParseError: If the DOT content is invalid or empty.
    """
    graphs = pydot.graph_from_dot_data(dot_content)
    if not graphs:
        raise ParseError("No graph found in DOT content")

    graph = graphs[0]
    pipeline_name = _unquote(graph.get_name()) or name

    node_defaults, edge_defaults = _extract_defaults(graph)
    nodes = _extract_nodes(graph, node_defaults)
    edges = _extract_edges(graph, edge_defaults)

    # Process subgraphs
    _extract_subgraphs(graph, nodes, edges, node_defaults, edge_defaults)

    start_node = _resolve_start(nodes)
    _mark_terminals(nodes, edges)

    graph_attrs = _graph_attrs(graph)

    return Pipeline(
        name=pipeline_name,
        nodes=nodes,
        edges=edges,
        start_node=start_node,
        metadata=graph_attrs,
        goal=str(graph_attrs.pop("goal", "")),
        default_max_retry=int(graph_attrs.pop("default_max_retry", 50)),
        retry_target=graph_attrs.pop("retry_target", None),
        fallback_retry_target=graph_attrs.pop("fallback_retry_target", None),
        default_fidelity=graph_attrs.pop("default_fidelity", None),
        model_stylesheet=graph_attrs.pop("model_stylesheet", None),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_defaults(graph: pydot.Dot) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract node [...] and edge [...] default blocks from the graph."""
    node_defaults: dict[str, Any] = {}
    edge_defaults: dict[str, Any] = {}

    for dot_node in graph.get_nodes():
        raw_name = _unquote(dot_node.get_name())
        if raw_name == "node":
            node_defaults = _clean_attrs(dot_node.obj_dict.get("attributes", {}))
        elif raw_name == "edge":
            edge_defaults = _clean_attrs(dot_node.obj_dict.get("attributes", {}))

    return node_defaults, edge_defaults


def _get_node_shape(dot_node: pydot.Node, defaults: dict[str, Any]) -> str:
    """Resolve the shape for a pydot node, checking multiple sources."""
    # Try get_shape() method first
    shape = dot_node.get_shape()
    if shape and shape != "":
        return _unquote(str(shape))

    # Try from obj_dict attributes
    attrs = dot_node.obj_dict.get("attributes", {})
    if "shape" in attrs:
        return _unquote(str(attrs["shape"]))

    # Fall back to defaults
    if "shape" in defaults:
        return _unquote(str(defaults["shape"]))

    return "box"


def _extract_nodes(
    graph: pydot.Dot,
    node_defaults: dict[str, Any],
) -> dict[str, PipelineNode]:
    nodes: dict[str, PipelineNode] = {}
    for dot_node in graph.get_nodes():
        raw_name = _unquote(dot_node.get_name())
        # Skip pseudo-nodes for graph defaults
        if raw_name in ("node", "edge", "graph", ""):
            continue

        node = _build_node(dot_node, raw_name, node_defaults)
        nodes[raw_name] = node

    return nodes


def _build_node(
    dot_node: pydot.Node,
    raw_name: str,
    node_defaults: dict[str, Any],
    extra_classes: list[str] | None = None,
) -> PipelineNode:
    """Build a PipelineNode from a pydot node."""
    # Merge defaults with node-specific attrs (node-specific wins)
    merged = dict(node_defaults)
    merged.update(_clean_attrs(dot_node.obj_dict.get("attributes", {})))

    shape = _get_node_shape(dot_node, node_defaults)
    merged.pop("shape", None)

    # Handler resolution: type attr > SHAPE_HANDLER_MAP > "codergen"
    explicit_type = merged.pop("type", None)
    if explicit_type is not None:
        handler_type = str(explicit_type)
    else:
        handler_type = SHAPE_HANDLER_MAP.get(shape, "codergen")

    is_start = handler_type == "start"
    is_terminal = handler_type == "exit"

    # Extract known fields
    label = str(merged.pop("label", raw_name))
    prompt = str(merged.pop("prompt", ""))
    max_retries = int(coerce_value(str(merged.pop("max_retries", "0"))))
    goal_gate = _to_bool(str(merged.pop("goal_gate", "false")))
    retry_target = merged.pop("retry_target", None)
    if retry_target is not None:
        retry_target = str(retry_target)
    fallback_retry_target = merged.pop("fallback_retry_target", None)
    if fallback_retry_target is not None:
        fallback_retry_target = str(fallback_retry_target)
    fidelity = merged.pop("fidelity", None)
    if fidelity is not None:
        fidelity = str(fidelity)
    thread_id = merged.pop("thread_id", None)
    if thread_id is not None:
        thread_id = str(thread_id)
    timeout_raw = merged.pop("timeout", None)
    timeout: float | None = None
    if timeout_raw is not None:
        timeout = parse_duration(str(timeout_raw))
    llm_model = merged.pop("llm_model", None)
    if llm_model is not None:
        llm_model = str(llm_model)
    llm_provider = merged.pop("llm_provider", None)
    if llm_provider is not None:
        llm_provider = str(llm_provider)
    reasoning_effort = str(merged.pop("reasoning_effort", "high"))
    auto_status = _to_bool(str(merged.pop("auto_status", "false")))
    allow_partial = _to_bool(str(merged.pop("allow_partial", "false")))

    # Classes: from class attr + extra_classes
    class_raw = merged.pop("class", None)
    classes: list[str] = []
    if class_raw is not None:
        classes = [c.strip() for c in str(class_raw).split(",") if c.strip()]
    if extra_classes:
        classes.extend(extra_classes)

    # Coerce remaining attributes
    remaining = {k: coerce_value(str(v)) for k, v in merged.items()}

    return PipelineNode(
        name=raw_name,
        handler_type=handler_type,
        attributes=remaining,
        is_start=is_start,
        is_terminal=is_terminal,
        label=label,
        shape=shape,
        classes=classes,
        prompt=prompt,
        max_retries=max_retries,
        goal_gate=goal_gate,
        retry_target=retry_target,
        fallback_retry_target=fallback_retry_target,
        fidelity=fidelity,
        thread_id=thread_id,
        timeout=timeout,
        llm_model=llm_model,
        llm_provider=llm_provider,
        reasoning_effort=reasoning_effort,
        auto_status=auto_status,
        allow_partial=allow_partial,
    )


def _extract_edges(
    graph: pydot.Dot,
    edge_defaults: dict[str, Any],
) -> list[PipelineEdge]:
    edges: list[PipelineEdge] = []
    for dot_edge in graph.get_edges():
        edge = _build_edge(dot_edge, edge_defaults)
        edges.append(edge)
    return edges


def _build_edge(
    dot_edge: pydot.Edge,
    edge_defaults: dict[str, Any],
) -> PipelineEdge:
    """Build a PipelineEdge from a pydot edge."""
    source = _unquote(dot_edge.get_source())
    target = _unquote(dot_edge.get_destination())

    # Merge defaults with edge-specific attrs
    merged = dict(edge_defaults)
    merged.update(_clean_attrs(dot_edge.obj_dict.get("attributes", {})))

    condition = merged.pop("condition", None)
    if condition is not None:
        condition = str(condition)
    label = str(merged.pop("label", ""))
    weight = int(coerce_value(str(merged.pop("weight", "0"))))
    fidelity = merged.pop("fidelity", None)
    if fidelity is not None:
        fidelity = str(fidelity)
    thread_id = merged.pop("thread_id", None)
    if thread_id is not None:
        thread_id = str(thread_id)
    loop_restart = _to_bool(str(merged.pop("loop_restart", "false")))

    return PipelineEdge(
        source=source,
        target=target,
        condition=condition,
        label=label,
        weight=weight,
        fidelity=fidelity,
        thread_id=thread_id,
        loop_restart=loop_restart,
    )


def _extract_subgraphs(
    graph: pydot.Dot,
    nodes: dict[str, PipelineNode],
    edges: list[PipelineEdge],
    node_defaults: dict[str, Any],
    edge_defaults: dict[str, Any],
) -> None:
    """Process subgraphs recursively, extracting nodes and edges."""
    for subgraph in graph.get_subgraphs():
        # Derive class from subgraph label
        sg_label = _unquote(subgraph.get_label() or subgraph.get_name() or "")
        # Strip "cluster_" prefix if present (common pydot convention)
        if sg_label.startswith("cluster_"):
            sg_label = sg_label[len("cluster_"):]
        derived_class = _derive_class(sg_label)

        # Extract subgraph-level defaults
        sg_node_defaults = dict(node_defaults)
        sg_edge_defaults = dict(edge_defaults)
        for dot_node in subgraph.get_nodes():
            raw_name = _unquote(dot_node.get_name())
            if raw_name == "node":
                sg_node_defaults.update(
                    _clean_attrs(dot_node.obj_dict.get("attributes", {}))
                )
            elif raw_name == "edge":
                sg_edge_defaults.update(
                    _clean_attrs(dot_node.obj_dict.get("attributes", {}))
                )

        # Extract nodes from subgraph
        extra_classes = [derived_class] if derived_class else []
        for dot_node in subgraph.get_nodes():
            raw_name = _unquote(dot_node.get_name())
            if raw_name in ("node", "edge", "graph", ""):
                continue
            node = _build_node(dot_node, raw_name, sg_node_defaults, extra_classes)
            nodes[raw_name] = node

        # Extract edges from subgraph
        for dot_edge in subgraph.get_edges():
            edge = _build_edge(dot_edge, sg_edge_defaults)
            edges.append(edge)

        # Recurse into nested subgraphs
        _extract_subgraphs(subgraph, nodes, edges, sg_node_defaults, sg_edge_defaults)


def _derive_class(label: str) -> str:
    """Derive a CSS-like class name from a subgraph label.

    Lowercase, replace spaces with hyphens, strip non-alphanumeric
    characters except hyphens.
    """
    name = label.lower().replace(" ", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)
    return name


def _resolve_start(nodes: dict[str, PipelineNode]) -> str:
    """Determine the start node name.

    Precedence:
    1. Node with ``handler_type == "start"`` (shape=Mdiamond)
    2. Node named ``"start"``
    3. Error
    """
    for node in nodes.values():
        if node.handler_type == "start":
            node.is_start = True
            return node.name

    if "start" in nodes:
        nodes["start"].is_start = True
        return "start"

    raise ParseError(
        "No start node found — use shape=Mdiamond or name a node 'start'"
    )


def _mark_terminals(nodes: dict[str, PipelineNode], edges: list[PipelineEdge]) -> None:
    """Mark terminal nodes.

    Nodes with handler_type == "exit" are terminal.
    Nodes with no outgoing edges are implicitly terminal.
    """
    sources = {e.source for e in edges}
    for node in nodes.values():
        if node.handler_type == "exit":
            node.is_terminal = True
        elif node.name not in sources and not node.is_terminal:
            node.is_terminal = True


def _graph_attrs(graph: pydot.Dot) -> dict[str, Any]:
    raw = graph.obj_dict.get("attributes", {})
    cleaned = _clean_attrs(raw)
    return {k: coerce_value(str(v)) for k, v in cleaned.items()}


def _clean_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Strip surrounding quotes from attribute values."""
    return {k: _unquote(str(v)) for k, v in attrs.items()}


def _unquote(value: str) -> str:
    """Remove surrounding double-quotes from a string."""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")
