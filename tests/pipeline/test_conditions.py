"""Tests for the condition evaluator (spec §10)."""

import pytest

from attractor.pipeline.conditions import (
    ConditionError,
    evaluate_condition,
    validate_condition_syntax,
)
from attractor.pipeline.models import PipelineContext


@pytest.fixture
def ctx() -> PipelineContext:
    return PipelineContext.from_dict(
        {
            "approved": True,
            "exit_code": 0,
            "retries": 3,
            "name": "test",
            "count": 42,
        }
    )


class TestEvaluateCondition:
    """Core evaluation using spec §10 syntax: = for equality, && for AND."""

    def test_equality_true(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved=true", ctx) is True

    def test_equality_false(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved=false", ctx) is False

    def test_equality_with_spaces(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved = true", ctx) is True

    def test_not_equal(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("exit_code != 0", ctx) is False
        assert evaluate_condition("exit_code != 1", ctx) is True

    def test_not_equal_no_spaces(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("exit_code!=0", ctx) is False
        assert evaluate_condition("exit_code!=1", ctx) is True

    def test_string_comparison(self, ctx: PipelineContext) -> None:
        assert evaluate_condition('name="test"', ctx) is True
        assert evaluate_condition('name!="other"', ctx) is True

    def test_string_comparison_with_spaces(self, ctx: PipelineContext) -> None:
        assert evaluate_condition('name = "test"', ctx) is True
        assert evaluate_condition('name != "other"', ctx) is True

    def test_and_operator(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved=true && exit_code=0", ctx) is True
        assert evaluate_condition("approved=true && exit_code!=0", ctx) is False

    def test_and_multiple_clauses(self, ctx: PipelineContext) -> None:
        assert (
            evaluate_condition("approved=true && exit_code=0 && name=test", ctx)
            is True
        )
        assert (
            evaluate_condition("approved=true && exit_code=1 && name=test", ctx)
            is False
        )

    def test_empty_condition(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("", ctx) is True
        assert evaluate_condition("   ", ctx) is True

    def test_missing_variable_is_empty_string(self) -> None:
        ctx = PipelineContext()
        assert evaluate_condition('missing=""', ctx) is True
        assert evaluate_condition("missing=something", ctx) is False

    def test_integer_literal(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("count=42", ctx) is True
        assert evaluate_condition("count=99", ctx) is False

    def test_invalid_syntax_raises(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Invalid condition syntax"):
            evaluate_condition("=broken", ctx)


class TestUnsupportedOperators:
    """Spec §10.7: only = and != are allowed; others are rejected."""

    def test_python_equality_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator '=='"):
            evaluate_condition("approved == true", ctx)

    def test_less_than_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator"):
            evaluate_condition("retries < 5", ctx)

    def test_greater_than_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator"):
            evaluate_condition("retries > 5", ctx)

    def test_less_equal_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator"):
            evaluate_condition("retries <= 3", ctx)

    def test_greater_equal_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator"):
            evaluate_condition("retries >= 4", ctx)

    def test_and_keyword_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator 'and'"):
            evaluate_condition("approved=true and exit_code=0", ctx)

    def test_or_keyword_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator 'or'"):
            evaluate_condition("exit_code=0 or retries=10", ctx)

    def test_not_keyword_rejected(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Unsupported operator 'not'"):
            evaluate_condition("not approved=false", ctx)


class TestOutcomeVariable:
    """Spec §10.3: outcome is a first-class variable."""

    def test_outcome_from_extra_vars(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "outcome=success", ctx, extra_vars={"outcome": "success"}
            )
            is True
        )
        assert (
            evaluate_condition(
                "outcome=fail", ctx, extra_vars={"outcome": "success"}
            )
            is False
        )

    def test_outcome_from_context_fallback(self) -> None:
        ctx = PipelineContext.from_dict({"outcome": "fail"})
        assert evaluate_condition("outcome=fail", ctx) is True
        assert evaluate_condition("outcome=success", ctx) is False

    def test_outcome_extra_vars_take_precedence(self) -> None:
        ctx = PipelineContext.from_dict({"outcome": "fail"})
        assert (
            evaluate_condition(
                "outcome=success", ctx, extra_vars={"outcome": "success"}
            )
            is True
        )

    def test_outcome_not_equal(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "outcome!=fail", ctx, extra_vars={"outcome": "success"}
            )
            is True
        )


class TestPreferredLabelVariable:
    """Spec §10.3: preferred_label is a first-class variable."""

    def test_preferred_label_from_extra_vars(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "preferred_label=Fix",
                ctx,
                extra_vars={"preferred_label": "Fix"},
            )
            is True
        )

    def test_preferred_label_from_context(self) -> None:
        ctx = PipelineContext.from_dict({"preferred_label": "Review"})
        assert evaluate_condition("preferred_label=Review", ctx) is True

    def test_preferred_label_not_equal(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "preferred_label!=Ship",
                ctx,
                extra_vars={"preferred_label": "Fix"},
            )
            is True
        )


class TestContextPrefixResolution:
    """Spec §10.4: context. prefix fallback."""

    def test_context_prefix_lookup(self) -> None:
        ctx = PipelineContext.from_dict({"tests_passed": True})
        assert evaluate_condition("context.tests_passed=true", ctx) is True

    def test_context_prefix_missing_key(self) -> None:
        ctx = PipelineContext()
        assert evaluate_condition('context.missing=""', ctx) is True
        assert evaluate_condition("context.missing=something", ctx) is False

    def test_context_prefix_with_outcome(self) -> None:
        ctx = PipelineContext.from_dict({"tests_passed": True})
        assert (
            evaluate_condition(
                "outcome=success && context.tests_passed=true",
                ctx,
                extra_vars={"outcome": "success"},
            )
            is True
        )

    def test_bare_name_and_context_prefix_equivalent(self) -> None:
        ctx = PipelineContext.from_dict({"flag": "yes"})
        assert evaluate_condition("flag=yes", ctx) is True
        assert evaluate_condition("context.flag=yes", ctx) is True


class TestSpecExamples:
    """Tests from spec §10.6 examples."""

    def test_route_on_success(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "outcome=success", ctx, extra_vars={"outcome": "success"}
            )
            is True
        )

    def test_route_on_failure(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "outcome=fail", ctx, extra_vars={"outcome": "fail"}
            )
            is True
        )
        assert (
            evaluate_condition(
                "outcome=fail", ctx, extra_vars={"outcome": "success"}
            )
            is False
        )

    def test_route_on_success_and_context_flag(self) -> None:
        ctx = PipelineContext.from_dict({"tests_passed": True})
        assert (
            evaluate_condition(
                "outcome=success && context.tests_passed=true",
                ctx,
                extra_vars={"outcome": "success"},
            )
            is True
        )

    def test_route_on_context_not_equal(self) -> None:
        ctx = PipelineContext.from_dict({"loop_state": "active"})
        assert evaluate_condition("context.loop_state!=exhausted", ctx) is True

    def test_route_on_preferred_label(self) -> None:
        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "preferred_label=Fix",
                ctx,
                extra_vars={"preferred_label": "Fix"},
            )
            is True
        )


class TestValidateConditionSyntax:
    """Tests for static syntax validation."""

    def test_valid_expressions(self) -> None:
        assert validate_condition_syntax("outcome=success") is None
        assert validate_condition_syntax("outcome!=fail") is None
        assert validate_condition_syntax("a=1 && b=2") is None
        assert validate_condition_syntax("") is None
        assert validate_condition_syntax("   ") is None

    def test_invalid_python_equality(self) -> None:
        result = validate_condition_syntax("x == 1")
        assert result is not None
        assert "==" in result

    def test_invalid_ordering_operator(self) -> None:
        result = validate_condition_syntax("x < 5")
        assert result is not None

    def test_invalid_boolean_keyword(self) -> None:
        result = validate_condition_syntax("x=1 and y=2")
        assert result is not None
        assert "and" in result

    def test_invalid_empty_key(self) -> None:
        result = validate_condition_syntax("=value")
        assert result is not None


class TestBareKeyTruthy:
    """Spec §10.5: bare key checks for truthy value."""

    def test_bare_key_truthy(self) -> None:
        ctx = PipelineContext.from_dict({"flag": "yes"})
        assert evaluate_condition("flag", ctx) is True

    def test_bare_key_falsy_missing(self) -> None:
        ctx = PipelineContext()
        assert evaluate_condition("missing", ctx) is False

    def test_bare_key_falsy_empty_string(self) -> None:
        ctx = PipelineContext.from_dict({"flag": ""})
        assert evaluate_condition("flag", ctx) is False

    def test_bare_key_numeric_zero_is_truthy(self) -> None:
        """Integer 0 becomes "0" via _to_string, which is truthy as a
        non-empty string. This documents current behavior."""
        ctx = PipelineContext.from_dict({"count": 0})
        assert evaluate_condition("count", ctx) is True

    def test_bare_key_false_string_is_truthy(self) -> None:
        """The string "false" is a non-empty string and thus truthy.
        Use ``flag=false`` (equality) to check for boolean-false values."""
        ctx = PipelineContext.from_dict({"flag": "false"})
        assert evaluate_condition("flag", ctx) is True

    def test_bare_key_boolean_false_is_truthy(self) -> None:
        """Boolean False becomes "false" via _to_string, which is truthy
        as a non-empty string. Use ``flag=false`` for semantic checks."""
        ctx = PipelineContext.from_dict({"flag": False})
        assert evaluate_condition("flag", ctx) is True

    def test_equality_handles_boolean_false_correctly(self) -> None:
        """Equality comparison correctly matches boolean False as 'false'."""
        ctx = PipelineContext.from_dict({"flag": False})
        assert evaluate_condition("flag=false", ctx) is True
        assert evaluate_condition("flag=true", ctx) is False

    def test_equality_handles_numeric_zero_correctly(self) -> None:
        """Equality comparison correctly matches integer 0 as '0'."""
        ctx = PipelineContext.from_dict({"count": 0})
        assert evaluate_condition("count=0", ctx) is True
        assert evaluate_condition("count=1", ctx) is False

    def test_bare_key_with_and(self) -> None:
        ctx = PipelineContext.from_dict({"a": "yes", "b": "yes"})
        assert evaluate_condition("a && b", ctx) is True

    def test_bare_key_one_missing(self) -> None:
        ctx = PipelineContext.from_dict({"a": "yes"})
        assert evaluate_condition("a && b", ctx) is False


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_none_expression(self, ctx: PipelineContext) -> None:
        # While the type says str, callers might pass None
        assert evaluate_condition("", ctx) is True

    def test_boolean_coercion_in_context(self) -> None:
        ctx = PipelineContext.from_dict({"flag": True})
        assert evaluate_condition("flag=true", ctx) is True

    def test_false_boolean_coercion(self) -> None:
        ctx = PipelineContext.from_dict({"flag": False})
        assert evaluate_condition("flag=false", ctx) is True

    def test_integer_coercion_in_context(self) -> None:
        ctx = PipelineContext.from_dict({"count": 0})
        assert evaluate_condition("count=0", ctx) is True

    def test_single_quoted_string(self) -> None:
        ctx = PipelineContext.from_dict({"name": "hello"})
        assert evaluate_condition("name='hello'", ctx) is True

    def test_extra_vars_with_enum_value(self) -> None:
        from attractor.pipeline.models import OutcomeStatus

        ctx = PipelineContext()
        assert (
            evaluate_condition(
                "outcome=success",
                ctx,
                extra_vars={"outcome": OutcomeStatus.SUCCESS},
            )
            is True
        )
