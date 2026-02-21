"""Tests for the DOT parser."""

import pytest

from attractor.pipeline.models import Pipeline
from attractor.pipeline.parser import ParseError, parse_dot_string


SIMPLE_DOT = """\
digraph simple {
    start [handler="codergen" prompt="Analyze code" start=true]
    done [handler="conditional" terminal=true]
    start -> done
}
"""

FULL_DOT = """\
digraph pipeline {
    start [handler="codergen" prompt="Analyze the codebase" start=true]
    review [handler="human_gate" prompt="Review the analysis"]
    implement [handler="codergen" prompt="Implement changes"]
    test [handler="tool" command="pytest -v"]
    done [handler="conditional" terminal=true]

    start -> review
    review -> implement [condition="approved == true"]
    review -> start [condition="approved == false" label="retry"]
    implement -> test
    test -> done [condition="exit_code == 0"]
    test -> implement [condition="exit_code != 0" label="fix"]
}
"""


class TestParseDotString:
    def test_simple_pipeline(self) -> None:
        pipeline = parse_dot_string(SIMPLE_DOT)
        assert pipeline.name == "simple"
        assert "start" in pipeline.nodes
        assert "done" in pipeline.nodes
        assert pipeline.start_node == "start"
        assert pipeline.nodes["start"].is_start is True
        assert pipeline.nodes["done"].is_terminal is True

    def test_node_attributes(self) -> None:
        pipeline = parse_dot_string(SIMPLE_DOT)
        start = pipeline.nodes["start"]
        assert start.handler_type == "codergen"
        assert start.attributes["prompt"] == "Analyze code"

    def test_full_pipeline_structure(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        assert len(pipeline.nodes) == 5
        assert len(pipeline.edges) == 6
        assert pipeline.start_node == "start"

    def test_edge_conditions(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        # Find the review -> implement edge
        edge = next(
            e
            for e in pipeline.edges
            if e.source == "review" and e.target == "implement"
        )
        assert edge.condition == "approved == true"

    def test_edge_labels(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        retry_edge = next(
            e
            for e in pipeline.edges
            if e.source == "review" and e.target == "start"
        )
        assert retry_edge.label == "retry"

    def test_implicit_terminal(self) -> None:
        """Nodes with no outgoing edges are implicitly terminal."""
        dot = """\
digraph g {
    a [handler="codergen" start=true]
            b [handler="tool" command="echo done"]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].is_terminal is True

    def test_start_by_name(self) -> None:
        """A node named 'start' is auto-detected as the start node."""
        dot = """\
digraph g {
    start [handler="codergen" prompt="begin"]
    end [handler="conditional" terminal=true]
    start -> end
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.start_node == "start"
        assert pipeline.nodes["start"].is_start is True

    def test_no_start_node_raises(self) -> None:
        dot = """\
digraph g {
    step1 [handler="codergen"]
    step2 [handler="tool" command="echo hi"]
    step1 -> step2
}
"""
        with pytest.raises(ParseError, match="No start node"):
            parse_dot_string(dot)

    def test_empty_dot_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_dot_string("")

    def test_handler_types_extracted(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        assert pipeline.nodes["start"].handler_type == "codergen"
        assert pipeline.nodes["review"].handler_type == "human_gate"
        assert pipeline.nodes["test"].handler_type == "tool"
        assert pipeline.nodes["done"].handler_type == "conditional"
