"""Pipeline execution engine.

Traverses a :class:`Pipeline` graph from its start node, dispatching
each node to its registered handler, evaluating outgoing edge conditions
for routing, and checkpointing state after every completed node.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.goals import GoalGate
from attractor.pipeline.handlers import (
    HandlerRegistry,
    create_default_registry,
)
from attractor.pipeline.models import (
    Checkpoint,
    NodeResult,
    OutcomeStatus,
    Pipeline,
    PipelineContext,
    PipelineEdge,
    PipelineNode,
)
from attractor.pipeline.state import save_checkpoint
from attractor.pipeline.stylesheet import ModelStylesheet, apply_stylesheet

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Raised for unrecoverable engine failures."""


class PipelineEngine:
    """Single-threaded pipeline execution engine.

    Walks the DAG from the start node.  For each node it:
    1. Resolves attributes via the stylesheet.
    2. Dispatches to the handler registered for the node's handler_type.
    3. Merges context_updates from the result.
    4. Evaluates outgoing edges (by priority) to pick the next node.
    5. Writes a checkpoint.

    Supports resume from a :class:`Checkpoint` and optional
    :class:`GoalGate` enforcement before allowing terminal exit.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        stylesheet: ModelStylesheet | None = None,
        checkpoint_dir: str | Path | None = None,
        goal_gate: GoalGate | None = None,
        max_steps: int = 1000,
    ) -> None:
        self._registry = registry
        self._stylesheet = stylesheet or ModelStylesheet()
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._goal_gate = goal_gate
        self._max_steps = max_steps

    async def run(
        self,
        pipeline: Pipeline,
        context: PipelineContext | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> PipelineContext:
        """Execute *pipeline* and return the final context.

        Args:
            pipeline: The pipeline definition to execute.
            context: Initial context (empty if ``None``).
            checkpoint: If provided, resume from this checkpoint state.

        Returns:
            The :class:`PipelineContext` after pipeline completion.

        Raises:
            EngineError: On unrecoverable errors (missing handler, etc.)
        """
        # Build a registry with pipeline awareness if none provided
        registry = self._registry or create_default_registry(pipeline=pipeline)

        if checkpoint:
            ctx = checkpoint.context
            current_node = checkpoint.current_node
            completed: list[str] = list(checkpoint.completed_nodes)
            logger.info(
                "Resuming pipeline '%s' from node '%s'",
                pipeline.name,
                current_node,
            )
        else:
            ctx = context or PipelineContext()
            current_node = pipeline.start_node
            completed = []

        if not current_node or current_node not in pipeline.nodes:
            raise EngineError(
                f"Start node '{current_node}' not found in pipeline '{pipeline.name}'"
            )

        # Set graph.goal at initialization
        if not ctx.has("graph.goal"):
            ctx.set("graph.goal", pipeline.goal)

        steps = 0
        while steps < self._max_steps:
            steps += 1
            node = pipeline.nodes[current_node]

            # Apply stylesheet defaults
            if self._stylesheet:
                node.attributes = apply_stylesheet(self._stylesheet, node)

            logger.info(
                "Executing node '%s' (handler: %s)", node.name, node.handler_type
            )

            # Dispatch to handler
            handler = registry.get(node.handler_type)
            if handler is None:
                raise EngineError(
                    f"No handler registered for type '{node.handler_type}' "
                    f"(node '{node.name}')"
                )

            # Retry loop
            max_attempts = node.max_retries + 1
            result: NodeResult | None = None

            for attempt in range(max_attempts):
                if attempt > 0:
                    delay = self._retry_delay(attempt, node)
                    logger.info(
                        "Retrying node '%s' (attempt %d/%d) after %.1fs",
                        node.name,
                        attempt + 1,
                        max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)

                result = await handler.execute(node, ctx)
                ctx.set(f"internal.retry_count.{node.name}", attempt)

                if result.status != OutcomeStatus.RETRY:
                    break

            assert result is not None  # At least one iteration always runs

            # Merge context updates
            if result.context_updates:
                ctx.update(result.context_updates)

            # Set spec-required context keys
            ctx.set("outcome", result.status.value)
            if result.preferred_label:
                ctx.set("preferred_label", result.preferred_label)
            ctx.set("current_node", node.name)
            ctx.set("last_stage", node.name)

            if result.status == OutcomeStatus.FAIL:
                logger.error(
                    "Node '%s' failed: %s", node.name, result.failure_reason
                )
                ctx.set("_last_error", result.failure_reason)
                ctx.set("_failed_node", node.name)

                # Failure routing per spec
                # 1. Check for edge with condition matching outcome==fail
                fail_edges = [
                    e
                    for e in pipeline.outgoing_edges(node.name)
                    if e.condition
                    and "outcome" in e.condition
                    and "fail" in e.condition
                ]
                routed = False
                if fail_edges:
                    ctx.set("outcome", "fail")
                    for edge in fail_edges:
                        try:
                            if edge.condition and evaluate_condition(edge.condition, ctx):
                                current_node = edge.target
                                routed = True
                                break
                        except Exception:
                            pass

                # 2. Use node's retry_target
                if not routed and node.retry_target and node.retry_target in pipeline.nodes:
                    current_node = node.retry_target
                    routed = True

                # 3. Use pipeline's fallback_retry_target
                if (
                    not routed
                    and node.fallback_retry_target
                    and node.fallback_retry_target in pipeline.nodes
                ):
                    current_node = node.fallback_retry_target
                    routed = True

                if routed:
                    completed.append(node.name)
                    self._save_checkpoint(
                        pipeline.name, current_node, ctx, completed
                    )
                    continue

            completed.append(node.name)

            # Determine next node
            next_node = self._resolve_next(result, node, pipeline, ctx)

            # Checkpoint with the *next* node so resume starts there
            self._save_checkpoint(
                pipeline.name,
                next_node if next_node else current_node,
                ctx,
                completed,
            )

            if next_node is None:
                # Potential terminal — check goal gates
                goal_gate_nodes = [
                    n for n in pipeline.nodes.values() if n.goal_gate
                ]
                if goal_gate_nodes:
                    completed_set = set(completed)
                    all_satisfied = all(
                        n.name in completed_set for n in goal_gate_nodes
                    )
                    if not all_satisfied:
                        unmet = [
                            n.name
                            for n in goal_gate_nodes
                            if n.name not in completed_set
                        ]
                        logger.warning("Goal gate not satisfied: %s", unmet)
                        ctx.set("_goal_gate_unmet", unmet)
                        # Route to pipeline retry_target
                        if (
                            pipeline.retry_target
                            and pipeline.retry_target in pipeline.nodes
                        ):
                            current_node = pipeline.retry_target
                            continue
                        if (
                            pipeline.fallback_retry_target
                            and pipeline.fallback_retry_target in pipeline.nodes
                        ):
                            current_node = pipeline.fallback_retry_target
                            continue

                # Also check legacy GoalGate
                if self._goal_gate and not self._goal_gate.check(completed, ctx):
                    unmet = self._goal_gate.unmet_requirements(completed, ctx)
                    logger.warning("Goal gate not satisfied: %s", unmet)
                    ctx.set("_goal_gate_unmet", unmet)

                logger.info(
                    "Pipeline '%s' completed at node '%s'",
                    pipeline.name,
                    node.name,
                )
                break

            if next_node not in pipeline.nodes:
                raise EngineError(f"Next node '{next_node}' does not exist in pipeline")

            current_node = next_node
        else:
            logger.warning(
                "Pipeline '%s' hit max steps (%d)", pipeline.name, self._max_steps
            )

        ctx.set("_completed_nodes", completed)
        return ctx

    async def run_sub_pipeline(
        self, pipeline_name: str, context: PipelineContext
    ) -> PipelineContext:
        """Hook for SupervisorHandler to run a named sub-pipeline.

        Subclasses or engine configurators should override this to
        resolve the pipeline by name and execute it.
        """
        logger.warning(
            "run_sub_pipeline called for '%s' but no resolver configured",
            pipeline_name,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_next(
        self,
        result: NodeResult,
        node: PipelineNode,
        pipeline: Pipeline,
        ctx: PipelineContext,
    ) -> str | None:
        """5-step edge selection per spec."""
        # Explicit routing override from handler
        if result.next_node:
            return result.next_node

        # Terminal node — stop
        if node.is_terminal:
            return None

        edges = pipeline.outgoing_edges(node.name)  # Already sorted by weight desc
        if not edges:
            return None  # Implicitly terminal

        # Step 1: Condition matching — collect eligible edges
        eligible: list[PipelineEdge] = []
        unconditional: list[PipelineEdge] = []
        for edge in edges:
            if edge.condition is None:
                unconditional.append(edge)
                eligible.append(edge)
            else:
                try:
                    if evaluate_condition(edge.condition, ctx):
                        eligible.append(edge)
                except Exception as exc:
                    logger.error(
                        "Error evaluating condition '%s': %s",
                        edge.condition,
                        exc,
                    )
                    ctx.set("_condition_error", str(exc))

        if not eligible:
            if unconditional:
                eligible = unconditional
            else:
                raise EngineError(
                    f"No edge matched for non-terminal node '{node.name}'"
                )

        # Step 2: Preferred label match
        if result.preferred_label:
            normalized = result.preferred_label.strip().lower()
            for edge in eligible:
                if edge.label.strip().lower() == normalized:
                    return edge.target

        # Step 3: Suggested next IDs
        if result.suggested_next_ids:
            for suggested_id in result.suggested_next_ids:
                for edge in eligible:
                    if edge.target == suggested_id:
                        return edge.target

        # Step 4: Highest weight (edges already sorted by weight desc)
        # Return the first eligible edge (highest weight)
        if eligible:
            # Among edges with equal weight, fall through to step 5
            max_weight = eligible[0].weight
            top_edges = [e for e in eligible if e.weight == max_weight]
            if len(top_edges) == 1:
                return top_edges[0].target

            # Step 5: Lexical tiebreak — target node ID first alphabetically
            top_edges.sort(key=lambda e: e.target)
            return top_edges[0].target

        return None

    def _retry_delay(self, attempt: int, node: PipelineNode) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base = 0.2  # 200ms initial
        factor = 2.0
        delay = base * (factor ** (attempt - 1))
        max_delay = 30.0
        delay = min(delay, max_delay)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0, delay + jitter)

    def _save_checkpoint(
        self,
        pipeline_name: str,
        current_node: str,
        ctx: PipelineContext,
        completed: list[str],
    ) -> None:
        if self._checkpoint_dir is None:
            return
        cp = Checkpoint(
            pipeline_name=pipeline_name,
            current_node=current_node,
            context=ctx,
            completed_nodes=list(completed),
            timestamp=time.time(),
        )
        path = save_checkpoint(cp, self._checkpoint_dir)
        logger.debug("Checkpoint saved to %s", path)
