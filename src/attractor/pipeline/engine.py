"""Pipeline execution engine.

Traverses a :class:`Pipeline` graph from its start node, dispatching
each node to its registered handler, evaluating outgoing edge conditions
for routing, and checkpointing state after every completed node.
"""

from __future__ import annotations

import logging
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
    Pipeline,
    PipelineContext,
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

            result = await handler.execute(node, ctx)

            # Merge context updates
            if result.context_updates:
                ctx.update(result.context_updates)

            if not result.success:
                logger.error("Node '%s' failed: %s", node.name, result.error)
                ctx.set("_last_error", result.error)
                ctx.set("_failed_node", node.name)

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
                # Potential terminal — check goal gate
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
        """Determine the next node to execute."""
        # Explicit routing override from the handler
        if result.next_node:
            return result.next_node

        # Terminal node — stop
        if node.is_terminal:
            return None

        # Evaluate outgoing edges in priority order
        edges = pipeline.outgoing_edges(node.name)
        default_target: str | None = None
        had_condition_error = False

        for edge in edges:
            if edge.condition is None:
                default_target = edge.target
                continue
            try:
                if evaluate_condition(edge.condition, ctx):
                    return edge.target
            except Exception as exc:
                had_condition_error = True
                logger.error(
                    "Error evaluating condition '%s': %s",
                    edge.condition,
                    exc,
                )
                ctx.set("_condition_error", str(exc))

        if default_target:
            return default_target

        if had_condition_error:
            raise EngineError(
                f"No edge matched for non-terminal node '{node.name}' "
                f"after condition evaluation error"
            )

        # No outgoing edges — implicitly terminal
        return None

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
