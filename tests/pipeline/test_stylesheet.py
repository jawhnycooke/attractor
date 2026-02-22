"""Tests for ModelStylesheet — rule-based default attributes for nodes."""

from attractor.pipeline.models import Pipeline, PipelineNode
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


class TestGraphLevelDefaults:
    """S3: Stylesheet resolution should consider graph-level default attributes."""

    def test_graph_defaults_used_when_no_rule_sets_value(self) -> None:
        """Graph-level defaults fill in values not set by any rule."""
        stylesheet = ModelStylesheet(
            rules=[StyleRule(temperature=0.5)]
        )
        node = _make_node()
        pipeline = Pipeline(
            name="test",
            metadata={"llm_model": "gpt-3.5", "custom_key": "from_graph"},
        )

        resolved = apply_stylesheet(stylesheet, node, pipeline=pipeline)
        assert resolved["llm_model"] == "gpt-3.5"
        assert resolved["custom_key"] == "from_graph"
        assert resolved["temperature"] == 0.5

    def test_stylesheet_rules_override_graph_defaults(self) -> None:
        """Stylesheet rules have higher priority than graph-level defaults."""
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_model="gpt-4o")]
        )
        node = _make_node()
        pipeline = Pipeline(
            name="test",
            metadata={"llm_model": "gpt-3.5"},
        )

        resolved = apply_stylesheet(stylesheet, node, pipeline=pipeline)
        assert resolved["llm_model"] == "gpt-4o"

    def test_node_attrs_override_graph_defaults(self) -> None:
        """Node attributes have higher priority than graph-level defaults."""
        stylesheet = ModelStylesheet(rules=[])
        node = _make_node(llm_model="claude-3-opus")
        pipeline = Pipeline(
            name="test",
            metadata={"llm_model": "gpt-3.5"},
        )

        resolved = apply_stylesheet(stylesheet, node, pipeline=pipeline)
        assert resolved["llm_model"] == "claude-3-opus"

    def test_full_cascade_order(self) -> None:
        """Cascade: graph defaults < stylesheet rules < node attributes."""
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(llm_model="gpt-4o", llm_provider="openai"),
            ]
        )
        # Node explicitly sets llm_provider via attributes
        node = _make_node(llm_provider="anthropic")
        pipeline = Pipeline(
            name="test",
            metadata={
                "llm_model": "gpt-3.5",
                "llm_provider": "default_provider",
                "timeout": 60,
            },
        )

        resolved = apply_stylesheet(stylesheet, node, pipeline=pipeline)
        # llm_model: graph(gpt-3.5) < rule(gpt-4o) → gpt-4o
        assert resolved["llm_model"] == "gpt-4o"
        # llm_provider: graph(default_provider) < rule(openai) < node(anthropic)
        assert resolved["llm_provider"] == "anthropic"
        # timeout: only graph(60) → 60
        assert resolved["timeout"] == 60

    def test_no_pipeline_parameter_works(self) -> None:
        """Backward compat: apply_stylesheet without pipeline still works."""
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_model="gpt-4o")]
        )
        node = _make_node()

        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "gpt-4o"


class TestNodeFieldSetting:
    """S5: apply_stylesheet should set PipelineNode fields directly."""

    def test_sets_llm_model_on_node(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_model="gpt-4o")]
        )
        node = _make_node()
        assert node.llm_model is None

        apply_stylesheet(stylesheet, node)
        assert node.llm_model == "gpt-4o"

    def test_sets_llm_provider_on_node(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_provider="openai")]
        )
        node = _make_node()
        assert node.llm_provider is None

        apply_stylesheet(stylesheet, node)
        assert node.llm_provider == "openai"

    def test_sets_reasoning_effort_on_node(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[StyleRule(reasoning_effort="low")]
        )
        node = _make_node()
        assert node.reasoning_effort == "high"  # default

        apply_stylesheet(stylesheet, node)
        assert node.reasoning_effort == "low"

    def test_does_not_override_existing_llm_model(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_model="gpt-4o")]
        )
        node = PipelineNode(
            name="my_node",
            handler_type="codergen",
            llm_model="claude-3-opus",
        )

        apply_stylesheet(stylesheet, node)
        assert node.llm_model == "claude-3-opus"

    def test_does_not_override_existing_llm_provider(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[StyleRule(llm_provider="openai")]
        )
        node = PipelineNode(
            name="my_node",
            handler_type="codergen",
            llm_provider="anthropic",
        )

        apply_stylesheet(stylesheet, node)
        assert node.llm_provider == "anthropic"

    def test_does_not_override_explicit_reasoning_effort(self) -> None:
        """Node attributes with reasoning_effort should not be overridden."""
        stylesheet = ModelStylesheet(
            rules=[StyleRule(reasoning_effort="low")]
        )
        node = _make_node(reasoning_effort="medium")

        apply_stylesheet(stylesheet, node)
        # reasoning_effort is in node.attributes, so should not be overridden
        assert node.reasoning_effort == "high"  # field stays default
        # But the attributes dict value is preserved in resolved output

    def test_sets_all_three_fields_together(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    llm_model="gpt-4o",
                    llm_provider="openai",
                    reasoning_effort="low",
                )
            ]
        )
        node = _make_node()

        resolved = apply_stylesheet(stylesheet, node)
        assert node.llm_model == "gpt-4o"
        assert node.llm_provider == "openai"
        assert node.reasoning_effort == "low"
        # Values should also be in the returned dict
        assert resolved["llm_model"] == "gpt-4o"
        assert resolved["llm_provider"] == "openai"
        assert resolved["reasoning_effort"] == "low"

    def test_fields_set_with_graph_defaults(self) -> None:
        """S3+S5: Graph-level defaults should also set node fields."""
        stylesheet = ModelStylesheet(rules=[])
        node = _make_node()
        pipeline = Pipeline(
            name="test",
            metadata={"llm_model": "gpt-3.5", "llm_provider": "openai"},
        )

        apply_stylesheet(stylesheet, node, pipeline=pipeline)
        assert node.llm_model == "gpt-3.5"
        assert node.llm_provider == "openai"
