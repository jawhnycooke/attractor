"""Tests for GoalGate — pipeline exit gate enforcement."""

from attractor.pipeline.goals import GoalGate
from attractor.pipeline.models import PipelineContext


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
            context_conditions=["tests_passed == true"],
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
            context_conditions=["approved == true"],
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
