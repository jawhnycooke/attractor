"""Tests for ModelStylesheet — rule-based default attributes for nodes."""

from attractor.pipeline.models import PipelineNode
from attractor.pipeline.stylesheet import (
    ModelStylesheet,
    StyleRule,
    apply_stylesheet,
)


def _make_node(
    name: str = "my_node",
    handler_type: str = "codergen",
    **attrs: object,
) -> PipelineNode:
    return PipelineNode(name=name, handler_type=handler_type, attributes=dict(attrs))


class TestStyleRuleMatching:
    def test_rule_matches_by_handler_type(self) -> None:
        rule = StyleRule(handler_type="codergen", model="gpt-4o")
        node_match = _make_node(handler_type="codergen")
        node_no_match = _make_node(handler_type="tool")

        assert rule.matches(node_match) is True
        assert rule.matches(node_no_match) is False

    def test_rule_matches_by_name_glob(self) -> None:
        rule = StyleRule(name_pattern="test_*", timeout=120.0)
        node_match = _make_node(name="test_unit")
        node_no_match = _make_node(name="build_step")

        assert rule.matches(node_match) is True
        assert rule.matches(node_no_match) is False


class TestStylesheetCascading:
    def test_stylesheet_cascading_later_wins(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(handler_type="codergen", model="gpt-3.5", temperature=0.5),
                StyleRule(handler_type="codergen", model="gpt-4o"),
            ]
        )
        node = _make_node(handler_type="codergen")

        resolved = apply_stylesheet(stylesheet, node)
        # Later rule's model should override the earlier one
        assert resolved["model"] == "gpt-4o"
        # Temperature from earlier rule persists since later didn't set it
        assert resolved["temperature"] == 0.5

    def test_node_attrs_override_stylesheet(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(handler_type="codergen", model="gpt-4o", temperature=0.7),
            ]
        )
        # Node explicitly sets model
        node = _make_node(handler_type="codergen", model="claude-3-opus")

        resolved = apply_stylesheet(stylesheet, node)
        # Node's own model attribute should win
        assert resolved["model"] == "claude-3-opus"
        # Stylesheet temperature should still apply
        assert resolved["temperature"] == 0.7
