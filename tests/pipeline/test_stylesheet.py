"""Tests for ModelStylesheet — rule-based default attributes for nodes."""

from attractor.pipeline.models import Pipeline, PipelineNode
from attractor.pipeline.stylesheet import (
    ModelStylesheet,
    StyleRule,
    apply_stylesheet,
    parse_stylesheet,
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


class TestShapeSelectorMatching:
    """P-P08: Shape selectors match nodes by their DOT shape attribute."""

    def test_shape_rule_matches_node_with_matching_shape(self) -> None:
        rule = StyleRule(
            selector_type="shape",
            selector_value="box",
            specificity=1,
            llm_model="gpt-4o",
        )
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        assert rule.matches(node) is True

    def test_shape_rule_does_not_match_different_shape(self) -> None:
        rule = StyleRule(
            selector_type="shape",
            selector_value="box",
            specificity=1,
            llm_model="gpt-4o",
        )
        node = PipelineNode(name="start", handler_type="start", shape="Mdiamond")
        assert rule.matches(node) is False

    def test_shape_rule_matches_mdiamond(self) -> None:
        rule = StyleRule(
            selector_type="shape",
            selector_value="Mdiamond",
            specificity=1,
            llm_model="gpt-4o",
        )
        node = PipelineNode(name="start", handler_type="start", shape="Mdiamond")
        assert rule.matches(node) is True

    def test_shape_rule_matches_msquare(self) -> None:
        rule = StyleRule(
            selector_type="shape",
            selector_value="Msquare",
            specificity=1,
            llm_model="gpt-4o",
        )
        node = PipelineNode(name="exit", handler_type="exit", shape="Msquare")
        assert rule.matches(node) is True

    def test_shape_rule_matches_hexagon(self) -> None:
        rule = StyleRule(
            selector_type="shape",
            selector_value="hexagon",
            specificity=1,
            llm_model="gpt-4o",
        )
        node = PipelineNode(name="gate", handler_type="wait.human", shape="hexagon")
        assert rule.matches(node) is True

    def test_shape_selector_applies_defaults_via_stylesheet(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="shape",
                    selector_value="box",
                    specificity=1,
                    llm_model="gpt-4o",
                    llm_provider="openai",
                ),
            ]
        )
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "gpt-4o"
        assert resolved["llm_provider"] == "openai"

    def test_shape_selector_does_not_apply_to_non_matching_shape(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="shape",
                    selector_value="hexagon",
                    specificity=1,
                    llm_model="gpt-4o",
                ),
            ]
        )
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert "llm_model" not in resolved


class TestShapeSelectorSpecificity:
    """P-P08: Specificity order is universal(0) < shape(1) < class(2) < ID(3)."""

    def test_shape_overrides_universal(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="universal",
                    selector_value="*",
                    specificity=0,
                    llm_model="universal-model",
                ),
                StyleRule(
                    selector_type="shape",
                    selector_value="box",
                    specificity=1,
                    llm_model="shape-model",
                ),
            ]
        )
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "shape-model"

    def test_class_overrides_shape(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="shape",
                    selector_value="box",
                    specificity=1,
                    llm_model="shape-model",
                ),
                StyleRule(
                    selector_type="class",
                    selector_value="fast",
                    specificity=2,
                    llm_model="class-model",
                ),
            ]
        )
        node = PipelineNode(
            name="impl", handler_type="codergen", shape="box", classes=["fast"]
        )
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "class-model"

    def test_id_overrides_class(self) -> None:
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="class",
                    selector_value="fast",
                    specificity=2,
                    llm_model="class-model",
                ),
                StyleRule(
                    selector_type="id",
                    selector_value="impl",
                    specificity=3,
                    llm_model="id-model",
                ),
            ]
        )
        node = PipelineNode(
            name="impl", handler_type="codergen", shape="box", classes=["fast"]
        )
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "id-model"

    def test_full_specificity_cascade(self) -> None:
        """All four selector types in one stylesheet — ID wins."""
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="universal",
                    specificity=0,
                    llm_model="universal-model",
                    llm_provider="universal-provider",
                    reasoning_effort="low",
                ),
                StyleRule(
                    selector_type="shape",
                    selector_value="box",
                    specificity=1,
                    llm_model="shape-model",
                    llm_provider="shape-provider",
                ),
                StyleRule(
                    selector_type="class",
                    selector_value="code",
                    specificity=2,
                    llm_model="class-model",
                ),
                StyleRule(
                    selector_type="id",
                    selector_value="review",
                    specificity=3,
                    llm_model="id-model",
                ),
            ]
        )
        node = PipelineNode(
            name="review", handler_type="codergen", shape="box", classes=["code"]
        )
        resolved = apply_stylesheet(stylesheet, node)
        # ID overrides class overrides shape overrides universal
        assert resolved["llm_model"] == "id-model"
        # llm_provider: shape overrides universal (class and ID didn't set it)
        assert resolved["llm_provider"] == "shape-provider"
        # reasoning_effort: only universal set it
        assert resolved["reasoning_effort"] == "low"

    def test_specificity_wins_regardless_of_rule_order(self) -> None:
        """Higher-specificity rules win even if declared before lower ones."""
        stylesheet = ModelStylesheet(
            rules=[
                StyleRule(
                    selector_type="id",
                    selector_value="my_node",
                    specificity=3,
                    llm_model="id-model",
                ),
                StyleRule(
                    selector_type="shape",
                    selector_value="box",
                    specificity=1,
                    llm_model="shape-model",
                ),
                StyleRule(
                    selector_type="universal",
                    specificity=0,
                    llm_model="universal-model",
                ),
            ]
        )
        node = PipelineNode(name="my_node", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "id-model"


class TestParseStylesheetShapeSelectors:
    """P-P08: parse_stylesheet recognizes bare-word selectors as shape selectors."""

    def test_parse_shape_selector(self) -> None:
        css = 'box { llm_model: gpt-4o; }'
        stylesheet = parse_stylesheet(css)
        assert len(stylesheet.rules) == 1
        rule = stylesheet.rules[0]
        assert rule.selector_type == "shape"
        assert rule.selector_value == "box"
        assert rule.specificity == 1
        assert rule.llm_model == "gpt-4o"

    def test_parse_mdiamond_shape_selector(self) -> None:
        css = 'Mdiamond { llm_model: claude-opus-4-6; }'
        stylesheet = parse_stylesheet(css)
        assert len(stylesheet.rules) == 1
        rule = stylesheet.rules[0]
        assert rule.selector_type == "shape"
        assert rule.selector_value == "Mdiamond"
        assert rule.specificity == 1

    def test_parse_mixed_selectors_with_shape(self) -> None:
        css = """
            * { llm_model: default-model; }
            box { llm_model: box-model; }
            .fast { llm_model: fast-model; }
            #review { llm_model: review-model; }
        """
        stylesheet = parse_stylesheet(css)
        assert len(stylesheet.rules) == 4

        # Verify selector types and specificity
        types = [(r.selector_type, r.specificity) for r in stylesheet.rules]
        assert types == [
            ("universal", 0),
            ("shape", 1),
            ("class", 2),
            ("id", 3),
        ]

    def test_parsed_shape_selector_matches_node(self) -> None:
        css = 'box { llm_model: gpt-4o; llm_provider: openai; }'
        stylesheet = parse_stylesheet(css)
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "gpt-4o"
        assert resolved["llm_provider"] == "openai"

    def test_parsed_shape_does_not_match_wrong_shape(self) -> None:
        css = 'hexagon { llm_model: gpt-4o; }'
        stylesheet = parse_stylesheet(css)
        node = PipelineNode(name="impl", handler_type="codergen", shape="box")
        resolved = apply_stylesheet(stylesheet, node)
        assert "llm_model" not in resolved

    def test_parsed_specificity_cascade_with_shape(self) -> None:
        """Full cascade via parse_stylesheet: universal < shape < class < ID."""
        css = """
            * { llm_model: universal; reasoning_effort: low; }
            box { llm_model: from-shape; llm_provider: shape-prov; }
            .code { llm_model: from-class; }
            #impl { llm_model: from-id; }
        """
        stylesheet = parse_stylesheet(css)
        node = PipelineNode(
            name="impl", handler_type="codergen", shape="box", classes=["code"]
        )
        resolved = apply_stylesheet(stylesheet, node)
        assert resolved["llm_model"] == "from-id"
        assert resolved["llm_provider"] == "shape-prov"
        assert resolved["reasoning_effort"] == "low"

    def test_parse_shape_selector_with_multiple_properties(self) -> None:
        css = 'diamond { llm_model: gpt-4o; llm_provider: openai; reasoning_effort: high; }'
        stylesheet = parse_stylesheet(css)
        assert len(stylesheet.rules) == 1
        rule = stylesheet.rules[0]
        assert rule.selector_type == "shape"
        assert rule.llm_model == "gpt-4o"
        assert rule.llm_provider == "openai"
        assert rule.reasoning_effort == "high"
