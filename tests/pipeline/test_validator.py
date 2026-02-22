"""Tests for the pipeline validator."""

import pytest

from attractor.pipeline.models import Pipeline, PipelineEdge, PipelineNode
from attractor.pipeline.validator import (
    ValidationError,
    ValidationException,
    ValidationLevel,
    has_errors,
    validate_or_raise,
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

    def test_unknown_handler_type_is_warning(self) -> None:
        """V4: Unknown handler type should be WARNING, not ERROR."""
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
        type_findings = [f for f in findings if "unknown_type" in f.message]
        assert len(type_findings) >= 1
        assert all(f.level == ValidationLevel.WARNING for f in type_findings)

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

    def test_unreachable_node_is_error(self) -> None:
        """V3: Unreachable nodes should be ERROR, not WARNING."""
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
        reachability_errors = [
            f
            for f in findings
            if f.node_name == "orphan" and f.rule == "reachability"
        ]
        assert len(reachability_errors) >= 1
        assert all(f.level == ValidationLevel.ERROR for f in reachability_errors)

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
        assert any("tool_command" in f.message for f in findings)

    def test_has_errors_helper(self) -> None:
        assert has_errors([ValidationError(level=ValidationLevel.ERROR, message="bad")])
        assert not has_errors(
            [ValidationError(level=ValidationLevel.WARNING, message="meh")]
        )
        assert not has_errors([])


class TestMultipleStartNodes:
    """V5: Exactly one start node required."""

    def test_multiple_start_nodes_error(self) -> None:
        nodes = {
            "s1": PipelineNode(name="s1", handler_type="start", is_start=True),
            "s2": PipelineNode(name="s2", handler_type="start", is_start=True),
            "end": PipelineNode(
                name="end", handler_type="exit", is_terminal=True
            ),
        }
        edges = [
            PipelineEdge(source="s1", target="end"),
            PipelineEdge(source="s2", target="end"),
        ]
        pipeline = _make_pipeline(
            nodes=nodes, edges=edges, start_node="s1"
        )
        findings = validate_pipeline(pipeline)
        assert any(
            f.level == ValidationLevel.ERROR
            and f.rule == "start_node"
            and "multiple" in f.message.lower()
            for f in findings
        )


class TestMultipleTerminalNodes:
    """V6: Exactly one terminal node required."""

    def test_multiple_terminal_nodes_error(self) -> None:
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="codergen", is_start=True
            ),
            "end1": PipelineNode(
                name="end1", handler_type="exit", is_terminal=True
            ),
            "end2": PipelineNode(
                name="end2", handler_type="exit", is_terminal=True
            ),
        }
        edges = [
            PipelineEdge(source="start", target="end1"),
            PipelineEdge(source="start", target="end2"),
        ]
        pipeline = _make_pipeline(
            nodes=nodes, edges=edges, start_node="start"
        )
        findings = validate_pipeline(pipeline)
        assert any(
            f.level == ValidationLevel.ERROR
            and f.rule == "terminal_node"
            and "multiple" in f.message.lower()
            for f in findings
        )


class TestStylesheetSyntax:
    """V1: stylesheet_syntax validation rule."""

    def test_valid_stylesheet_no_error(self) -> None:
        pipeline = _make_pipeline(
            model_stylesheet='* { llm_model: gpt-4o; }'
        )
        findings = validate_pipeline(pipeline)
        assert not any(f.rule == "stylesheet_syntax" for f in findings)

    def test_no_stylesheet_no_error(self) -> None:
        pipeline = _make_pipeline()
        findings = validate_pipeline(pipeline)
        assert not any(f.rule == "stylesheet_syntax" for f in findings)

    def test_empty_stylesheet_error(self) -> None:
        pipeline = _make_pipeline(model_stylesheet="not a valid stylesheet")
        findings = validate_pipeline(pipeline)
        stylesheet_findings = [f for f in findings if f.rule == "stylesheet_syntax"]
        assert len(stylesheet_findings) >= 1
        assert all(f.level == ValidationLevel.ERROR for f in stylesheet_findings)


class TestFidelityValid:
    """V2: fidelity_valid validation rule."""

    def test_valid_fidelity_no_warning(self) -> None:
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="codergen", is_start=True,
                fidelity="full",
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True,
                fidelity="compact",
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        findings = validate_pipeline(pipeline)
        assert not any(f.rule == "fidelity_valid" for f in findings)

    def test_invalid_node_fidelity_warning(self) -> None:
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="codergen", is_start=True,
                fidelity="invalid_mode",
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True,
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        findings = validate_pipeline(pipeline)
        fidelity_findings = [f for f in findings if f.rule == "fidelity_valid"]
        assert len(fidelity_findings) >= 1
        assert all(f.level == ValidationLevel.WARNING for f in fidelity_findings)

    def test_invalid_default_fidelity_warning(self) -> None:
        pipeline = _make_pipeline(default_fidelity="bad_mode")
        findings = validate_pipeline(pipeline)
        fidelity_findings = [f for f in findings if f.rule == "fidelity_valid"]
        assert len(fidelity_findings) >= 1
        assert all(f.level == ValidationLevel.WARNING for f in fidelity_findings)

    def test_invalid_edge_fidelity_warning(self) -> None:
        edges = [PipelineEdge(source="start", target="end", fidelity="wrong")]
        pipeline = _make_pipeline(edges=edges)
        findings = validate_pipeline(pipeline)
        fidelity_findings = [f for f in findings if f.rule == "fidelity_valid"]
        assert len(fidelity_findings) >= 1
        assert all(f.level == ValidationLevel.WARNING for f in fidelity_findings)

    def test_valid_summary_fidelity_modes(self) -> None:
        """All summary:* modes should be accepted."""
        for mode in ("summary:low", "summary:medium", "summary:high"):
            nodes = {
                "start": PipelineNode(
                    name="start", handler_type="codergen", is_start=True,
                    fidelity=mode,
                ),
                "end": PipelineNode(
                    name="end", handler_type="conditional", is_terminal=True,
                ),
            }
            pipeline = _make_pipeline(nodes=nodes)
            findings = validate_pipeline(pipeline)
            assert not any(f.rule == "fidelity_valid" for f in findings), (
                f"Valid fidelity mode '{mode}' should not produce a warning"
            )


class TestValidateOrRaise:
    """V7: validate_or_raise convenience function."""

    def test_valid_pipeline_returns_findings(self) -> None:
        pipeline = _make_pipeline()
        findings = validate_or_raise(pipeline)
        assert isinstance(findings, list)
        # No errors, so no exception raised
        assert not has_errors(findings)

    def test_error_pipeline_raises(self) -> None:
        pipeline = _make_pipeline(start_node="")
        with pytest.raises(ValidationException) as exc_info:
            validate_or_raise(pipeline)
        assert len(exc_info.value.errors) >= 1
        assert all(
            e.level == ValidationLevel.ERROR for e in exc_info.value.errors
        )

    def test_warning_only_does_not_raise(self) -> None:
        """Warnings should not cause an exception."""
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="unknown_type", is_start=True
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        findings = validate_or_raise(pipeline)
        # type_known is WARNING, so no exception
        assert any(f.rule == "type_known" for f in findings)


class TestStylesheetAppliedBeforeValidation:
    """S4: Stylesheet should be applied as a pre-validation transform."""

    def test_stylesheet_applied_sets_node_fields(self) -> None:
        """Validator should apply stylesheet to nodes before running checks."""
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="codergen", is_start=True
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        }
        pipeline = _make_pipeline(
            nodes=nodes,
            model_stylesheet="* { llm_model: gpt-4o; }",
        )
        # Before validation, llm_model is not set
        assert pipeline.nodes["start"].llm_model is None

        validate_pipeline(pipeline)

        # After validation, stylesheet should have been applied
        assert pipeline.nodes["start"].llm_model == "gpt-4o"
        assert pipeline.nodes["end"].llm_model == "gpt-4o"

    def test_invalid_stylesheet_does_not_block_validation(self) -> None:
        """Invalid stylesheet should not prevent other validation checks."""
        pipeline = _make_pipeline(
            model_stylesheet="not a valid stylesheet at all",
        )
        findings = validate_pipeline(pipeline)
        # Should still report stylesheet_syntax error
        assert any(f.rule == "stylesheet_syntax" for f in findings)
        # But other checks should still have run (e.g., start_node is valid)
        assert not any(f.rule == "start_node" for f in findings)

    def test_no_stylesheet_skips_application(self) -> None:
        """No stylesheet means no pre-validation step."""
        nodes = {
            "start": PipelineNode(
                name="start", handler_type="codergen", is_start=True
            ),
            "end": PipelineNode(
                name="end", handler_type="conditional", is_terminal=True
            ),
        }
        pipeline = _make_pipeline(nodes=nodes)
        validate_pipeline(pipeline)
        # No stylesheet to apply, so llm_model stays None
        assert pipeline.nodes["start"].llm_model is None


class TestLintRuleName:
    """V8: LintRule protocol has name attribute."""

    def test_lint_rule_protocol_has_name(self) -> None:
        from attractor.pipeline.validator import LintRule

        class MyRule:
            name: str = "my_rule"

            def check(self, pipeline: Pipeline) -> list:
                return []

        rule = MyRule()
        assert isinstance(rule, LintRule)
        assert rule.name == "my_rule"
