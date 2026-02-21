"""DOT file parser for pipeline definitions.

Uses the ``pydot`` library to parse GraphViz DOT files into
:class:`~attractor.pipeline.models.Pipeline` objects.  Node and edge
attributes are extracted and mapped onto the pipeline data model.

A node's ``handler`` attribute selects its handler type.  The ``start``
and ``terminal`` boolean attributes mark entry and exit points.  If no
node has ``start=true``, the node named ``"start"`` is used.  Nodes
with no outgoing edges are implicitly terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydot

from attractor.pipeline.models import Pipeline, PipelineEdge, PipelineNode


class ParseError(Exception):
    """Raised when a DOT file cannot be parsed into a valid pipeline."""


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

    nodes = _extract_nodes(graph)
    edges = _extract_edges(graph)

    start_node = _resolve_start(nodes)
    _mark_terminals(nodes, edges)

    return Pipeline(
        name=pipeline_name,
        nodes=nodes,
        edges=edges,
        start_node=start_node,
        metadata=_graph_attrs(graph),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_nodes(graph: pydot.Dot) -> dict[str, PipelineNode]:
    nodes: dict[str, PipelineNode] = {}
    for dot_node in graph.get_nodes():
        raw_name = _unquote(dot_node.get_name())
        # pydot includes pseudo-nodes for graph defaults; skip them
        if raw_name in ("node", "edge", "graph", ""):
            continue

        attrs = _clean_attrs(dot_node.obj_dict.get("attributes", {}))
        handler_type = attrs.pop("handler", "codergen")
        is_start = _to_bool(attrs.pop("start", "false"))
        is_terminal = _to_bool(attrs.pop("terminal", "false"))

        nodes[raw_name] = PipelineNode(
            name=raw_name,
            handler_type=handler_type,
            attributes=attrs,
            is_start=is_start,
            is_terminal=is_terminal,
        )

    return nodes


def _extract_edges(graph: pydot.Dot) -> list[PipelineEdge]:
    edges: list[PipelineEdge] = []
    for dot_edge in graph.get_edges():
        source = _unquote(dot_edge.get_source())
        target = _unquote(dot_edge.get_destination())
        attrs = _clean_attrs(dot_edge.obj_dict.get("attributes", {}))

        condition = attrs.pop("condition", None)
        label = attrs.pop("label", "")
        priority = int(attrs.pop("priority", "0"))

        edges.append(PipelineEdge(
            source=source,
            target=target,
            condition=condition,
            label=label,
            priority=priority,
        ))

    return edges


def _resolve_start(nodes: dict[str, PipelineNode]) -> str:
    """Determine the start node name.

    Precedence:
    1. Node with ``is_start=True``
    2. Node named ``"start"``
    3. Error
    """
    for node in nodes.values():
        if node.is_start:
            return node.name

    if "start" in nodes:
        nodes["start"].is_start = True
        return "start"

    raise ParseError("No start node found — set start=true on a node or name one 'start'")


def _mark_terminals(
    nodes: dict[str, PipelineNode], edges: list[PipelineEdge]
) -> None:
    """Implicitly mark nodes with no outgoing edges as terminal."""
    sources = {e.source for e in edges}
    for node in nodes.values():
        if node.name not in sources and not node.is_terminal:
            node.is_terminal = True


def _graph_attrs(graph: pydot.Dot) -> dict[str, Any]:
    raw = graph.obj_dict.get("attributes", {})
    return _clean_attrs(raw)


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
