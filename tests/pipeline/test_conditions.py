"""Tests for the condition evaluator."""

import pytest

from attractor.pipeline.conditions import ConditionError, evaluate_condition
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
    def test_equality_true(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved == true", ctx) is True

    def test_equality_false(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved == false", ctx) is False

    def test_not_equal(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("exit_code != 0", ctx) is False
        assert evaluate_condition("exit_code != 1", ctx) is True

    def test_numeric_comparison(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("retries < 5", ctx) is True
        assert evaluate_condition("retries > 5", ctx) is False
        assert evaluate_condition("retries <= 3", ctx) is True
        assert evaluate_condition("retries >= 4", ctx) is False

    def test_string_comparison(self, ctx: PipelineContext) -> None:
        assert evaluate_condition('name == "test"', ctx) is True
        assert evaluate_condition('name != "other"', ctx) is True

    def test_and_operator(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("approved == true and exit_code == 0", ctx) is True
        assert evaluate_condition("approved == true and exit_code != 0", ctx) is False

    def test_or_operator(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("exit_code == 0 or retries > 10", ctx) is True

    def test_not_operator(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("not approved == false", ctx) is True

    def test_empty_condition(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("", ctx) is True
        assert evaluate_condition("   ", ctx) is True

    def test_missing_variable_is_none(self) -> None:
        ctx = PipelineContext()
        assert evaluate_condition("missing == none", ctx) is True

    def test_integer_literal(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("count == 42", ctx) is True

    def test_invalid_syntax_raises(self, ctx: PipelineContext) -> None:
        with pytest.raises(ConditionError, match="Invalid condition syntax"):
            evaluate_condition("== broken ==", ctx)

    def test_chained_comparison(self, ctx: PipelineContext) -> None:
        assert evaluate_condition("1 < retries < 5", ctx) is True
        assert evaluate_condition("5 < retries < 10", ctx) is False
