"""Tests for the pipeline validator."""

from attractor.pipeline.models import Pipeline, PipelineEdge, PipelineNode
from attractor.pipeline.validator import (
    ValidationLevel,
    has_errors,
    validate_pipeline,
)


def _make_pipeline(**overrides) -> Pipeline:
    """Helper to create a minimal valid pipeline."""
    defaults = {
        "name": "test",
        "nodes": {
            "start": PipelineNode(name="start", handler_type="codergen", is_start=True),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        },
        "edges": [PipelineEdge(source="start", target="end")],
        "start_node": "start",
        "metadata": {},
    }
    defaults.update(overrides)
    return Pipeline(**defaults)


class TestValidatePipeline:
    def test_valid_pipeline_no_errors(self) -> None:
        pipeline = _make_pipeline()
        findings = validate_pipeline(pipeline)
        assert not has_errors(findings)

    def test_no_start_node(self) -> None:
        pipeline = _make_pipeline(start_node="")
        findings = validate_pipeline(pipeline)
        assert has_errors(findings)
        assert any("start node" in f.message.lower() for f in findings)

    def test_missing_start_node_reference(self) -> None:
        pipeline = _make_pipeline(start_node="nonexistent")
        findings = validate_pipeline(pipeline)
        assert has_errors(findings)

    def test_no_terminal_nodes(self) -> None:
        nodes = {
            "a": PipelineNode(name="a", handler_type="codergen", is_start=True),
            "b": PipelineNode(name="b", handler_type="codergen"),
        }
        edges = [
            PipelineEdge(source="a", target="b"),
            PipelineEdge(source="b", target="a"),
        ]
        pipeline = _make_pipeline(nodes=nodes, edges=edges)
        findings = validate_pipeline(pipeline)
        assert any(
            f.level == ValidationLevel.ERROR and "terminal" in f.message.lower()
            for f in findings
        )

    def test_unknown_handler_type(self) -> None:
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="unknown_type", is_start=True
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        findings = validate_pipeline(pipeline)
        assert any("unknown_type" in f.message for f in findings)

    def test_edge_references_nonexistent_source(self) -> None:
        edges = [PipelineEdge(source="ghost", target="end")]
        pipeline = _make_pipeline(edges=edges)
        findings = validate_pipeline(pipeline)
        assert any("ghost" in f.message for f in findings)

    def test_edge_references_nonexistent_target(self) -> None:
        edges = [PipelineEdge(source="start", target="ghost")]
        pipeline = _make_pipeline(edges=edges)
        findings = validate_pipeline(pipeline)
        assert any("ghost" in f.message for f in findings)

    def test_invalid_condition_syntax(self) -> None:
        edges = [PipelineEdge(source="start", target="end", condition="== broken ==")]
        pipeline = _make_pipeline(edges=edges)
        findings = validate_pipeline(pipeline)
        assert any("condition" in f.message.lower() for f in findings)

    def test_unreachable_node_warning(self) -> None:
        nodes = {
            "start": PipelineNode(name="start", handler_type="codergen", is_start=True),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
            "orphan": PipelineNode(name="orphan", handler_type="codergen"),
        }
        edges = [PipelineEdge(source="start", target="end")]
        pipeline = _make_pipeline(nodes=nodes, edges=edges)
        findings = validate_pipeline(pipeline)
        warnings = [
            f
            for f in findings
            if f.level == ValidationLevel.WARNING and f.node_name == "orphan"
        ]
        assert len(warnings) >= 1

    def test_missing_required_tool_attribute(self) -> None:
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="tool", is_start=True, attributes={}
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        findings = validate_pipeline(pipeline)
        assert any("command" in f.message for f in findings)

    def test_has_errors_helper(self) -> None:
        from attractor.pipeline.validator import ValidationError

        assert has_errors([ValidationError(level=ValidationLevel.ERROR, message="bad")])
        assert not has_errors(
            [ValidationError(level=ValidationLevel.WARNING, message="meh")]
        )
        assert not has_errors([])
