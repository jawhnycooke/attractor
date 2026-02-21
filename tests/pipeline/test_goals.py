"""Tests for GoalGate — pipeline exit gate enforcement."""

import pytest

from attractor.pipeline.goals import GoalGate
from attractor.pipeline.models import (
    Pipeline,
    PipelineContext,
    PipelineEdge,
    PipelineNode,
)


class TestGoalGate:
    def test_goal_gate_requires_specific_nodes(self) -> None:
        gate = GoalGate(required_nodes=["build", "test", "deploy"])
        ctx = PipelineContext()

        # Only build and test completed — gate should fail
        assert gate.check(["build", "test"], ctx) is False
        # All three completed — gate should pass
        assert gate.check(["build", "test", "deploy"], ctx) is True

    def test_goal_gate_requires_context_condition(self) -> None:
        gate = GoalGate(
            required_nodes=[],
            context_conditions=["tests_passed=true"],
        )

        # Condition fails
        ctx_fail = PipelineContext.from_dict({"tests_passed": False})
        assert gate.check([], ctx_fail) is False

        # Condition passes
        ctx_pass = PipelineContext.from_dict({"tests_passed": True})
        assert gate.check([], ctx_pass) is True

    def test_unmet_requirements_lists_specifics(self) -> None:
        gate = GoalGate(
            required_nodes=["build", "test"],
            context_conditions=["approved=true"],
        )
        ctx = PipelineContext.from_dict({"approved": False})

        issues = gate.unmet_requirements(["build"], ctx)
        # Should list the missing node and the failing condition
        assert len(issues) == 2
        assert any("test" in i for i in issues)
        assert any("approved" in i for i in issues)

    def test_empty_goal_gate_always_passes(self) -> None:
        gate = GoalGate()
        ctx = PipelineContext()

        assert gate.check([], ctx) is True
        assert gate.unmet_requirements([], ctx) == []


class TestGoalGateIntegration:
    """Integration tests verifying GoalGate blocks terminal exit via engine."""

    @pytest.mark.asyncio
    async def test_engine_sets_goal_gate_unmet_context(self) -> None:
        """When a GoalGate is unsatisfied at a terminal node, the engine
        should set _goal_gate_unmet in context."""
        from attractor.pipeline.engine import PipelineEngine

        gate = GoalGate(
            required_nodes=["build"],
            context_conditions=["tests_passed=true"],
        )

        pipeline = Pipeline(
            name="test",
            start_node="start",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="start", is_start=True
                ),
                "done": PipelineNode(
                    name="done", handler_type="exit", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="done")],
        )

        engine = PipelineEngine(goal_gate=gate)
        ctx = await engine.run(pipeline)

        # GoalGate should have flagged the unmet requirements
        unmet = ctx.get("_goal_gate_unmet")
        assert unmet is not None
        assert len(unmet) == 2  # missing "build" node + failing condition
        assert any("build" in str(u) for u in unmet)
        assert any("tests_passed" in str(u) for u in unmet)
