"""Goal gate enforcement for pipeline exits.

A :class:`GoalGate` specifies a set of nodes that must have completed
successfully—and optionally context conditions that must hold—before
the engine is allowed to exit through a terminal node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.models import PipelineContext


@dataclass
class GoalGate:
    """Gate that must be satisfied before pipeline completion.

    Attributes:
        required_nodes: Node names that must appear in the completed set.
        context_conditions: Optional condition expressions that must all
            evaluate to ``True`` against the final context.
    """

    required_nodes: list[str] = field(default_factory=list)
    context_conditions: list[str] = field(default_factory=list)

    def check(
        self,
        completed_nodes: list[str],
        context: PipelineContext,
    ) -> bool:
        """Return ``True`` if all required nodes completed and conditions hold."""
        completed_set = set(completed_nodes)
        for node_name in self.required_nodes:
            if node_name not in completed_set:
                return False

        for condition in self.context_conditions:
            if not evaluate_condition(condition, context):
                return False

        return True

    def unmet_requirements(
        self,
        completed_nodes: list[str],
        context: PipelineContext,
    ) -> list[str]:
        """Return human-readable descriptions of unmet requirements."""
        issues: list[str] = []
        completed_set = set(completed_nodes)

        for node_name in self.required_nodes:
            if node_name not in completed_set:
                issues.append(f"Required node '{node_name}' has not completed")

        for condition in self.context_conditions:
            if not evaluate_condition(condition, context):
                issues.append(f"Context condition not met: {condition}")

        return issues
