"""Tests for the DOT parser."""

import pytest

from attractor.pipeline.parser import (
    SHAPE_HANDLER_MAP,
    ParseError,
    _derive_class,
    parse_dot_string,
)

# Uses shape-based handler resolution:
# Mdiamond = start, Msquare = exit, box = codergen (default)
SIMPLE_DOT = """\
digraph simple {
    begin [shape=Mdiamond prompt="Analyze code"]
    done [shape=Msquare]
    begin -> done
}
"""

FULL_DOT = """\
digraph pipeline {
    begin [shape=Mdiamond prompt="Analyze the codebase"]
    review [shape=hexagon prompt="Review the analysis"]
    implement [shape=box prompt="Implement changes"]
    test [shape=parallelogram]
    done [shape=Msquare]

    begin -> review
    review -> implement [condition="approved=true"]
    review -> begin [condition="approved=false" label="retry"]
    implement -> test
    test -> done [condition="exit_code=0"]
    test -> implement [condition="exit_code!=0" label="fix"]
}
"""


class TestParseDotString:
    def test_simple_pipeline(self) -> None:
        pipeline = parse_dot_string(SIMPLE_DOT)
        assert pipeline.name == "simple"
        assert "begin" in pipeline.nodes
        assert "done" in pipeline.nodes
        assert pipeline.start_node == "begin"
        assert pipeline.nodes["begin"].is_start is True
        assert pipeline.nodes["done"].is_terminal is True

    def test_node_shape_extraction(self) -> None:
        pipeline = parse_dot_string(SIMPLE_DOT)
        begin = pipeline.nodes["begin"]
        assert begin.shape == "Mdiamond"
        assert begin.handler_type == "start"
        assert begin.prompt == "Analyze code"

    def test_full_pipeline_structure(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        assert len(pipeline.nodes) == 5
        assert len(pipeline.edges) == 6
        assert pipeline.start_node == "begin"

    def test_edge_conditions(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        edge = next(
            e
            for e in pipeline.edges
            if e.source == "review" and e.target == "implement"
        )
        assert edge.condition == "approved=true"

    def test_edge_labels(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        retry_edge = next(
            e for e in pipeline.edges if e.source == "review" and e.target == "begin"
        )
        assert retry_edge.label == "retry"

    def test_implicit_terminal(self) -> None:
        """Nodes with no outgoing edges are implicitly terminal."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].is_terminal is True

    def test_start_by_name(self) -> None:
        """A node named 'start' is auto-detected as the start node."""
        dot = """\
digraph g {
    start [prompt="begin"]
    end [shape=Msquare]
    start -> end
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.start_node == "start"
        assert pipeline.nodes["start"].is_start is True

    def test_no_start_node_raises(self) -> None:
        dot = """\
digraph g {
    step1 [shape=box]
    step2 [shape=parallelogram]
    step1 -> step2
}
"""
        with pytest.raises(ParseError, match="No start node"):
            parse_dot_string(dot)

    def test_empty_dot_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_dot_string("")

    def test_handler_types_from_shapes(self) -> None:
        pipeline = parse_dot_string(FULL_DOT)
        assert pipeline.nodes["begin"].handler_type == "start"
        assert pipeline.nodes["review"].handler_type == "wait.human"
        assert pipeline.nodes["test"].handler_type == "tool"
        assert pipeline.nodes["done"].handler_type == "exit"

    def test_type_attr_overrides_shape(self) -> None:
        """Explicit type attribute overrides shape-based handler."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box type="custom_handler"]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].handler_type == "custom_handler"

    def test_default_shape_is_box(self) -> None:
        """Nodes without explicit shape default to box / codergen."""
        dot = """\
digraph g {
    start [shape=Mdiamond]
    worker []
    start -> worker
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["worker"].shape == "box"
        assert pipeline.nodes["worker"].handler_type == "codergen"

    def test_exit_node_is_terminal(self) -> None:
        """Nodes with shape=Msquare (handler=exit) are terminal."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=Msquare]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].is_terminal is True
        assert pipeline.nodes["b"].handler_type == "exit"


class TestEdgeWeight:
    def test_weight_attribute(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    c [shape=box]
    a -> b [weight=10]
    a -> c [weight=5]
}
"""
        pipeline = parse_dot_string(dot)
        outgoing = pipeline.outgoing_edges("a")
        assert outgoing[0].weight == 10
        assert outgoing[1].weight == 5

    def test_edge_fidelity_and_thread_id(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    a -> b [fidelity="high" thread_id="main"]
}
"""
        pipeline = parse_dot_string(dot)
        edge = pipeline.edges[0]
        assert edge.fidelity == "high"
        assert edge.thread_id == "main"

    def test_edge_loop_restart(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    b -> a [loop_restart=true]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        restart_edge = next(e for e in pipeline.edges if e.source == "b")
        assert restart_edge.loop_restart is True


class TestNodeFields:
    def test_node_known_fields(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    work [shape=box prompt="Do the work" max_retries=3 goal_gate=true
          retry_target="a" fidelity="standard" thread_id="t1"
          timeout="15m" llm_model="gpt-4o" llm_provider="openai"
          reasoning_effort="medium" auto_status=true allow_partial=true]
    a -> work
}
"""
        pipeline = parse_dot_string(dot)
        node = pipeline.nodes["work"]
        assert node.prompt == "Do the work"
        assert node.max_retries == 3
        assert node.goal_gate is True
        assert node.retry_target == "a"
        assert node.fidelity == "standard"
        assert node.thread_id == "t1"
        assert node.timeout == 900.0  # 15m
        assert node.llm_model == "gpt-4o"
        assert node.llm_provider == "openai"
        assert node.reasoning_effort == "medium"
        assert node.auto_status is True
        assert node.allow_partial is True

    def test_node_classes(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box class="fast,critical"]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].classes == ["fast", "critical"]

    def test_label_defaults_to_name(self) -> None:
        dot = """\
digraph g {
    mynode [shape=Mdiamond]
    other [shape=box]
    mynode -> other
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["mynode"].label == "mynode"

    def test_explicit_label(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond label="Start Node"]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["a"].label == "Start Node"

    def test_unknown_attrs_remain_in_attributes(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond custom_key="custom_val"]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["a"].attributes["custom_key"] == "custom_val"

    def test_attribute_coercion(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond custom_int="42" custom_bool="true" custom_float="3.14"]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["a"].attributes["custom_int"] == 42
        assert pipeline.nodes["a"].attributes["custom_bool"] is True
        assert pipeline.nodes["a"].attributes["custom_float"] == 3.14


class TestGraphAttrs:
    def test_graph_level_fields(self) -> None:
        dot = """\
digraph g {
    goal="Fix all bugs"
    default_max_retry=10
    retry_target="repair"
    default_fidelity="high"
    model_stylesheet="styles.yaml"

    start [shape=Mdiamond]
    repair [shape=box]
    start -> repair
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.goal == "Fix all bugs"
        assert pipeline.default_max_retry == 10
        assert pipeline.retry_target == "repair"
        assert pipeline.default_fidelity == "high"
        assert pipeline.model_stylesheet == "styles.yaml"


class TestNodeDefaults:
    def test_node_default_block(self) -> None:
        """node [...] default block applies to all nodes."""
        dot = """\
digraph g {
    node [llm_model="claude-3"]
    a [shape=Mdiamond]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["a"].llm_model == "claude-3"
        assert pipeline.nodes["b"].llm_model == "claude-3"

    def test_node_specific_overrides_default(self) -> None:
        """Node-specific attrs override defaults."""
        dot = """\
digraph g {
    node [llm_model="claude-3"]
    a [shape=Mdiamond llm_model="gpt-4o"]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["a"].llm_model == "gpt-4o"
        assert pipeline.nodes["b"].llm_model == "claude-3"

    def test_edge_default_block(self) -> None:
        """edge [...] default block applies to all edges."""
        dot = """\
digraph g {
    edge [fidelity="standard"]
    a [shape=Mdiamond]
    b [shape=box]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.edges[0].fidelity == "standard"


class TestSubgraphs:
    def test_subgraph_nodes_extracted(self) -> None:
        dot = """\
digraph g {
    start [shape=Mdiamond]
    subgraph cluster_backend {
        label="Backend Tasks"
        api [shape=box]
        db [shape=box]
        api -> db
    }
    start -> api
}
"""
        pipeline = parse_dot_string(dot)
        assert "api" in pipeline.nodes
        assert "db" in pipeline.nodes

    def test_subgraph_class_derived(self) -> None:
        dot = """\
digraph g {
    start [shape=Mdiamond]
    subgraph cluster_backend {
        label="Backend Tasks"
        api [shape=box]
    }
    start -> api
}
"""
        pipeline = parse_dot_string(dot)
        assert "backend-tasks" in pipeline.nodes["api"].classes

    def test_subgraph_edges_extracted(self) -> None:
        dot = """\
digraph g {
    start [shape=Mdiamond]
    subgraph cluster_work {
        label="Work"
        a [shape=box]
        b [shape=box]
        a -> b
    }
    start -> a
}
"""
        pipeline = parse_dot_string(dot)
        sub_edge = next(
            (e for e in pipeline.edges if e.source == "a" and e.target == "b"), None
        )
        assert sub_edge is not None

    def test_subgraph_defaults(self) -> None:
        dot = """\
digraph g {
    start [shape=Mdiamond]
    subgraph cluster_llm {
        label="LLM Work"
        node [llm_model="claude-3"]
        task1 [shape=box]
        task2 [shape=box]
    }
    start -> task1
    task1 -> task2
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["task1"].llm_model == "claude-3"
        assert pipeline.nodes["task2"].llm_model == "claude-3"


class TestDeriveClass:
    def test_basic(self) -> None:
        assert _derive_class("Backend Tasks") == "backend-tasks"

    def test_special_chars_stripped(self) -> None:
        assert _derive_class("My @Task! #1") == "my-task-1"

    def test_empty(self) -> None:
        assert _derive_class("") == ""

    def test_already_clean(self) -> None:
        assert _derive_class("simple") == "simple"


class TestShapeHandlerMap:
    def test_all_shapes_mapped(self) -> None:
        """Verify all expected shapes are in the map."""
        expected = {
            "Mdiamond",
            "Msquare",
            "box",
            "hexagon",
            "diamond",
            "component",
            "tripleoctagon",
            "parallelogram",
            "house",
        }
        assert set(SHAPE_HANDLER_MAP.keys()) == expected

    def test_unknown_shape_falls_back(self) -> None:
        """Unknown shapes fall back to codergen."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=ellipse]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["b"].handler_type == "codergen"


class TestRejectUndirectedAndStrict:
    """P1: Reject undirected graphs and strict modifier."""

    def test_undirected_graph_raises(self) -> None:
        dot = """\
graph g {
    a -- b
}
"""
        with pytest.raises(ParseError, match="[Dd]irected"):
            parse_dot_string(dot)

    def test_strict_digraph_raises(self) -> None:
        dot = """\
strict digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    a -> b
}
"""
        with pytest.raises(ParseError, match="strict"):
            parse_dot_string(dot)


class TestRejectMultipleGraphs:
    """P2: Reject DOT files containing multiple graphs."""

    def test_multiple_graphs_raises(self) -> None:
        dot = """\
digraph g1 {
    a [shape=Mdiamond]
    b [shape=box]
    a -> b
}
digraph g2 {
    c [shape=Mdiamond]
    d [shape=box]
    c -> d
}
"""
        with pytest.raises(ParseError, match="[Mm]ultiple"):
            parse_dot_string(dot)


class TestChainedEdges:
    """P3: Test coverage for chained edges (a -> b -> c syntax)."""

    def test_chained_edge_creates_multiple_edges(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    c [shape=Msquare]
    a -> b -> c
}
"""
        pipeline = parse_dot_string(dot)
        ab = next(
            (e for e in pipeline.edges if e.source == "a" and e.target == "b"),
            None,
        )
        bc = next(
            (e for e in pipeline.edges if e.source == "b" and e.target == "c"),
            None,
        )
        assert ab is not None
        assert bc is not None

    def test_chained_edge_with_attributes(self) -> None:
        """Attributes on chained edges apply to all edges in the chain."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    c [shape=Msquare]
    a -> b -> c [label="next"]
}
"""
        pipeline = parse_dot_string(dot)
        ab = next(
            (e for e in pipeline.edges if e.source == "a" and e.target == "b"),
            None,
        )
        bc = next(
            (e for e in pipeline.edges if e.source == "b" and e.target == "c"),
            None,
        )
        assert ab is not None
        assert bc is not None
        assert ab.label == "next"
        assert bc.label == "next"

    def test_chained_edge_three_hops(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    c [shape=box]
    d [shape=Msquare]
    a -> b -> c -> d
}
"""
        pipeline = parse_dot_string(dot)
        assert len(pipeline.edges) == 3
        ab = next(
            (e for e in pipeline.edges if e.source == "a" and e.target == "b"),
            None,
        )
        bc = next(
            (e for e in pipeline.edges if e.source == "b" and e.target == "c"),
            None,
        )
        cd = next(
            (e for e in pipeline.edges if e.source == "c" and e.target == "d"),
            None,
        )
        assert ab is not None
        assert bc is not None
        assert cd is not None

    def test_chained_edge_count(self) -> None:
        """Chain of N nodes produces exactly N-1 edges."""
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=box]
    c [shape=Msquare]
    a -> b -> c
}
"""
        pipeline = parse_dot_string(dot)
        assert len(pipeline.edges) == 2


class TestGraphLabel:
    """P4: Extract graph-level label attribute."""

    def test_graph_label_extracted(self) -> None:
        dot = """\
digraph g {
    label="My Pipeline"
    a [shape=Mdiamond]
    b [shape=Msquare]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.label == "My Pipeline"

    def test_graph_label_default_empty(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b [shape=Msquare]
    a -> b
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.label == ""


class TestCaseInsensitiveStartNode:
    """P5: Case-insensitive start node detection."""

    def test_start_lowercase(self) -> None:
        dot = """\
digraph g {
    start [prompt="begin"]
    done [shape=Msquare]
    start -> done
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.start_node == "start"

    def test_start_uppercase(self) -> None:
        dot = """\
digraph g {
    START [prompt="begin"]
    done [shape=Msquare]
    START -> done
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.start_node == "START"

    def test_start_mixed_case(self) -> None:
        dot = """\
digraph g {
    Start [prompt="begin"]
    done [shape=Msquare]
    Start -> done
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.start_node == "Start"


class TestCaseInsensitiveTerminal:
    """P6: Terminal detection for 'exit' and 'end' names (case-insensitive)."""

    def test_exit_lowercase_is_terminal(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    exit [shape=box]
    a -> exit
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["exit"].is_terminal is True

    def test_exit_uppercase_is_terminal(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    EXIT [shape=box]
    a -> EXIT
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["EXIT"].is_terminal is True

    def test_end_lowercase_is_terminal(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    end [shape=box]
    a -> end
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["end"].is_terminal is True

    def test_end_uppercase_is_terminal(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    END [shape=box]
    a -> END
}
"""
        pipeline = parse_dot_string(dot)
        assert pipeline.nodes["END"].is_terminal is True


class TestLexicalTiebreak:
    """P7: outgoing_edges tiebreak lexically by target node name when weights equal."""

    def test_same_weight_lexical_order(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    c_node [shape=box]
    b_node [shape=box]
    a -> c_node [weight=5]
    a -> b_node [weight=5]
}
"""
        pipeline = parse_dot_string(dot)
        outgoing = pipeline.outgoing_edges("a")
        assert len(outgoing) == 2
        # Same weight, so lexical order: b_node < c_node
        assert outgoing[0].target == "b_node"
        assert outgoing[1].target == "c_node"

    def test_different_weight_overrides_lexical(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    b_node [shape=box]
    c_node [shape=box]
    a -> b_node [weight=1]
    a -> c_node [weight=10]
}
"""
        pipeline = parse_dot_string(dot)
        outgoing = pipeline.outgoing_edges("a")
        # Higher weight first
        assert outgoing[0].target == "c_node"
        assert outgoing[1].target == "b_node"

    def test_zero_weight_lexical_tiebreak(self) -> None:
        dot = """\
digraph g {
    a [shape=Mdiamond]
    z_node [shape=box]
    a_node [shape=box]
    m_node [shape=box]
    a -> z_node
    a -> a_node
    a -> m_node
}
"""
        pipeline = parse_dot_string(dot)
        outgoing = pipeline.outgoing_edges("a")
        targets = [e.target for e in outgoing]
        assert targets == ["a_node", "m_node", "z_node"]
