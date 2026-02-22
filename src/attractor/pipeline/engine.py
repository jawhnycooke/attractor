"""Pipeline execution engine.

Traverses a :class:`Pipeline` graph from its start node, dispatching
each node to its registered handler, evaluating outgoing edge conditions
for routing, and checkpointing state after every completed node.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from pathlib import Path

from attractor.pipeline.artifact_store import LocalArtifactStore
from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.events import PipelineEvent, PipelineEventEmitter, PipelineEventType
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
    RetryPolicy,
)
from attractor.pipeline.state import save_checkpoint
from attractor.pipeline.stylesheet import ModelStylesheet, apply_stylesheet
from attractor.pipeline.transforms import TransformRegistry

# Valid fidelity modes per spec §5.4
VALID_FIDELITY_MODES = frozenset({
    "full", "truncate", "compact",
    "summary:low", "summary:medium", "summary:high",
})

# Max value length in bytes before truncation in "truncate" mode
_TRUNCATE_MAX_VALUE_LEN = 500

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Raised for unrecoverable engine failures."""


def create_stage_dir(logs_root: Path, node_name: str) -> Path:
    """Create a standardized stage directory for a pipeline node.

    Creates ``{logs_root}/{node_name}/`` with the directory
    structure required by the run directory spec.

    Args:
        logs_root: Root directory for pipeline run logs.
        node_name: Name of the pipeline node.

    Returns:
        Path to the created stage directory.
    """
    stage_dir = logs_root / node_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


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
        event_emitter: PipelineEventEmitter | None = None,
        logs_root: str | Path | None = None,
        default_retry_policy: RetryPolicy | None = None,
        transform_registry: TransformRegistry | None = None,
        artifact_store: LocalArtifactStore | None = None,
    ) -> None:
        self._registry = registry
        self._stylesheet = stylesheet or ModelStylesheet()
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._goal_gate = goal_gate
        self._max_steps = max_steps
        self._event_emitter = event_emitter
        self._logs_root: Path | None = Path(logs_root) if logs_root else None
        self._default_retry_policy = default_retry_policy
        self._transform_registry = transform_registry
        self._artifact_store = artifact_store

    async def _emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event if an emitter is configured."""
        if self._event_emitter is not None:
            await self._event_emitter.emit(event)

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
        # Apply transforms between parse and validate (spec §9.1)
        if self._transform_registry:
            pipeline = self._transform_registry.apply_all(pipeline)

        # Build a registry with pipeline awareness if none provided
        registry = self._registry or create_default_registry(pipeline=pipeline)

        # Create artifact store per pipeline run (spec §5.5)
        artifact_store = self._artifact_store or LocalArtifactStore(
            base_dir=self._logs_root,
        )

        if checkpoint:
            ctx = checkpoint.context
            current_node = checkpoint.current_node
            completed: list[str] = list(checkpoint.completed_nodes)
            node_outcomes: dict[str, OutcomeStatus] = {}
            logger.info(
                "Resuming pipeline '%s' from node '%s'",
                pipeline.name,
                current_node,
            )
        else:
            ctx = context or PipelineContext()
            current_node = pipeline.start_node
            completed = []
            node_outcomes = {}

        prev_node: str | None = None
        start_time = time.time()

        if not current_node or current_node not in pipeline.nodes:
            raise EngineError(
                f"Start node '{current_node}' not found in pipeline '{pipeline.name}'"
            )

        # Set graph.goal at initialization
        if not ctx.has("graph.goal"):
            ctx.set("graph.goal", pipeline.goal)

        # Emit pipeline start event
        await self._emit(PipelineEvent(
            type=PipelineEventType.PIPELINE_START,
            pipeline_name=pipeline.name,
        ))

        steps = 0
        while steps < self._max_steps:
            steps += 1
            node = pipeline.nodes[current_node]

            # Apply stylesheet defaults
            if self._stylesheet:
                node.attributes = apply_stylesheet(self._stylesheet, node)

            # Resolve and apply fidelity mode (spec §5.4)
            incoming_edge = self._find_incoming_edge(
                current_node, pipeline, prev_node
            )
            resolved_fidelity = self._resolve_fidelity(
                incoming_edge, node, pipeline
            )
            self._apply_fidelity(ctx, resolved_fidelity)

            # E3: Check goal gates BEFORE executing terminal node handler
            if node.is_terminal:
                goal_gate_nodes = [
                    n for n in pipeline.nodes.values() if n.goal_gate
                ]
                if goal_gate_nodes:
                    unsatisfied = self._check_goal_gates(
                        goal_gate_nodes, node_outcomes
                    )
                    if unsatisfied is not None:
                        logger.warning(
                            "Goal gate not satisfied: %s", unsatisfied.name
                        )
                        ctx.set("_goal_gate_unmet", [unsatisfied.name])
                        # E10: Try retry targets in order
                        retry_target = self._find_retry_target(
                            unsatisfied, pipeline
                        )
                        if retry_target:
                            current_node = retry_target
                            continue
                        # E11: No retry target — emit PIPELINE_FAILED and raise
                        error_msg = (
                            f"Goal gate '{unsatisfied.name}' unsatisfied "
                            f"and no retry target found"
                        )
                        duration = time.time() - start_time
                        await self._emit(PipelineEvent(
                            type=PipelineEventType.PIPELINE_FAILED,
                            pipeline_name=pipeline.name,
                            data={"error": error_msg, "duration": duration},
                        ))
                        raise EngineError(error_msg)

                # Legacy GoalGate check
                if self._goal_gate and not self._goal_gate.check(completed, ctx):
                    unmet = self._goal_gate.unmet_requirements(completed, ctx)
                    logger.warning("Goal gate not satisfied: %s", unmet)
                    ctx.set("_goal_gate_unmet", unmet)

            logger.info(
                "Executing node '%s' (handler: %s)", node.name, node.handler_type
            )

            # Emit node start event
            await self._emit(PipelineEvent(
                type=PipelineEventType.NODE_START,
                node_name=node.name,
                pipeline_name=pipeline.name,
                data={"handler_type": node.handler_type},
            ))

            # Dispatch to handler
            handler = registry.get(node.handler_type)
            if handler is None:
                error_msg = (
                    f"No handler registered for type '{node.handler_type}' "
                    f"(node '{node.name}')"
                )
                duration = time.time() - start_time
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PIPELINE_FAILED,
                    pipeline_name=pipeline.name,
                    data={"error": error_msg, "duration": duration},
                ))
                raise EngineError(error_msg)

            # Determine max_attempts: retry_policy → engine default → max_retries
            if node.retry_policy:
                max_attempts = node.retry_policy.max_attempts
            elif self._default_retry_policy and node.max_retries == 0:
                max_attempts = self._default_retry_policy.max_attempts
            else:
                # E4: Inherit max_retries from pipeline when node has 0/unset
                effective_max_retries = node.max_retries
                if effective_max_retries == 0:
                    effective_max_retries = pipeline.metadata.get(
                        "default_max_retry", 0
                    )
                max_attempts = effective_max_retries + 1
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
                    # Emit node retry event
                    await self._emit(PipelineEvent(
                        type=PipelineEventType.NODE_RETRY,
                        node_name=node.name,
                        pipeline_name=pipeline.name,
                        data={"attempt": attempt + 1, "max_attempts": max_attempts, "delay": delay},
                    ))
                    await asyncio.sleep(delay)

                # E5: Catch handler exceptions and treat as retriable
                try:
                    if node.timeout is not None:
                        result = await asyncio.wait_for(
                            handler.execute(node, ctx, pipeline, self._logs_root),
                            timeout=node.timeout,
                        )
                    else:
                        result = await handler.execute(node, ctx, pipeline, self._logs_root)
                except asyncio.TimeoutError:
                    logger.error(
                        "Node '%s' timed out after %ss",
                        node.name,
                        node.timeout,
                    )
                    result = NodeResult(
                        status=OutcomeStatus.FAIL,
                        failure_reason=f"Node timed out after {node.timeout}s",
                    )
                    ctx.set("_last_error", result.failure_reason)
                    ctx.set("_failed_node", node.name)
                    break
                except Exception as exc:
                    logger.error(
                        "Handler exception for node '%s': %s", node.name, exc
                    )
                    if attempt < max_attempts - 1:
                        continue
                    result = NodeResult(
                        status=OutcomeStatus.FAIL,
                        failure_reason=str(exc),
                    )
                    break

                ctx.set(f"internal.retry_count.{node.name}", attempt)

                # E5: On SUCCESS/PARTIAL_SUCCESS break immediately
                if result.status in (
                    OutcomeStatus.SUCCESS,
                    OutcomeStatus.PARTIAL_SUCCESS,
                ):
                    break

                # E5: On RETRY with exhausted retries, check allow_partial
                if result.status == OutcomeStatus.RETRY:
                    if attempt < max_attempts - 1:
                        continue
                    if node.allow_partial:
                        result = NodeResult(
                            status=OutcomeStatus.PARTIAL_SUCCESS,
                            output=result.output,
                            context_updates=result.context_updates,
                            notes="retries exhausted, partial accepted",
                        )
                    else:
                        result = NodeResult(
                            status=OutcomeStatus.FAIL,
                            failure_reason="max retries exceeded",
                            output=result.output,
                            context_updates=result.context_updates,
                        )
                    break

                # On FAIL, break immediately (no retry for explicit FAIL)
                if result.status == OutcomeStatus.FAIL:
                    break

                # SKIPPED or other — break
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

            # E14: SKIPPED status — skip outcome recording
            if result.status != OutcomeStatus.SKIPPED:
                completed.append(node.name)
                # E9: Track node outcomes
                node_outcomes[node.name] = result.status

            # Emit node complete or fail event
            if result.status == OutcomeStatus.FAIL:
                await self._emit(PipelineEvent(
                    type=PipelineEventType.NODE_FAIL,
                    node_name=node.name,
                    pipeline_name=pipeline.name,
                    data={"failure_reason": result.failure_reason},
                ))
            else:
                await self._emit(PipelineEvent(
                    type=PipelineEventType.NODE_COMPLETE,
                    node_name=node.name,
                    pipeline_name=pipeline.name,
                    data={"status": result.status.value},
                ))

            # E7: Failure routing — do NOT fall through to normal edge selection
            if result.status == OutcomeStatus.FAIL:
                logger.error(
                    "Node '%s' failed: %s", node.name, result.failure_reason
                )
                ctx.set("_last_error", result.failure_reason)
                ctx.set("_failed_node", node.name)

                extra_vars = {
                    "outcome": result.status.value,
                    "preferred_label": result.preferred_label or "",
                }

                # 1. Check for fail edge (condition matching outcome=fail)
                routed = False
                for edge in pipeline.outgoing_edges(node.name):
                    if (
                        edge.condition
                        and "outcome" in edge.condition
                        and "fail" in edge.condition
                    ):
                        try:
                            if evaluate_condition(
                                edge.condition, ctx, extra_vars=extra_vars
                            ):
                                current_node = edge.target
                                routed = True
                                break
                        except Exception:
                            pass

                # 2. Node retry_target
                if (
                    not routed
                    and node.retry_target
                    and node.retry_target in pipeline.nodes
                ):
                    current_node = node.retry_target
                    routed = True

                # 3. Node fallback_retry_target
                if (
                    not routed
                    and node.fallback_retry_target
                    and node.fallback_retry_target in pipeline.nodes
                ):
                    current_node = node.fallback_retry_target
                    routed = True

                if routed:
                    await self._save_checkpoint_and_emit(
                        pipeline.name, current_node, ctx, completed
                    )
                    continue

                # E7/E13: No route found — emit PIPELINE_FAILED and raise
                duration = time.time() - start_time
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PIPELINE_FAILED,
                    pipeline_name=pipeline.name,
                    data={
                        "error": f"Node '{node.name}' failed with no failure route: "
                                 f"{result.failure_reason}",
                        "duration": duration,
                    },
                ))
                raise EngineError(
                    f"Node '{node.name}' failed with no failure route: "
                    f"{result.failure_reason}"
                )

            # Determine next node
            next_node = self._resolve_next(result, node, pipeline, ctx)

            # E12: Check loop_restart on the selected edge
            if next_node is not None:
                selected_edge = self._find_edge(
                    node.name, next_node, pipeline
                )
                if selected_edge and selected_edge.loop_restart:
                    completed = []
                    node_outcomes = {}
                    current_node = next_node
                    await self._save_checkpoint_and_emit(
                        pipeline.name, current_node, ctx, completed
                    )
                    continue

            # Checkpoint with the *next* node so resume starts there
            await self._save_checkpoint_and_emit(
                pipeline.name,
                next_node if next_node else current_node,
                ctx,
                completed,
            )

            if next_node is None:
                logger.info(
                    "Pipeline '%s' completed at node '%s'",
                    pipeline.name,
                    node.name,
                )
                # Emit pipeline complete event
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PIPELINE_COMPLETE,
                    pipeline_name=pipeline.name,
                ))
                break

            if next_node not in pipeline.nodes:
                error_msg = f"Next node '{next_node}' does not exist in pipeline"
                duration = time.time() - start_time
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PIPELINE_FAILED,
                    pipeline_name=pipeline.name,
                    data={"error": error_msg, "duration": duration},
                ))
                raise EngineError(error_msg)

            prev_node = current_node
            current_node = next_node
        else:
            logger.warning(
                "Pipeline '%s' hit max steps (%d)",
                pipeline.name,
                self._max_steps,
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

    @staticmethod
    def _resolve_fidelity(
        incoming_edge: PipelineEdge | None,
        target_node: PipelineNode,
        pipeline: Pipeline,
    ) -> str:
        """Resolve fidelity mode using 4-level precedence (spec §5.4).

        Precedence (highest to lowest):
        1. Edge ``fidelity`` attribute (on the incoming edge)
        2. Target node ``fidelity`` attribute
        3. Graph ``default_fidelity`` attribute
        4. Default: ``compact``

        Args:
            incoming_edge: The edge leading to *target_node*, or ``None``
                for the start node.
            target_node: The node about to execute.
            pipeline: The pipeline definition.

        Returns:
            The resolved fidelity mode string.
        """
        # 1. Edge fidelity (highest precedence)
        if incoming_edge and incoming_edge.fidelity:
            return incoming_edge.fidelity

        # 2. Node fidelity
        if target_node.fidelity:
            return target_node.fidelity

        # 3. Graph default_fidelity
        if pipeline.default_fidelity:
            return pipeline.default_fidelity

        # 4. Fallback
        return "compact"

    @staticmethod
    def _apply_fidelity(ctx: PipelineContext, fidelity: str) -> None:
        """Apply the resolved fidelity mode to the pipeline context.

        Records the mode in ``_fidelity_mode`` and transforms context
        data according to the mode:

        - ``full``: No transformation; all context data preserved.
        - ``truncate``: Trim string values exceeding the truncation limit.
        - ``compact``: Remove large values, keep only small scalars.
        - ``summary:*``: Mark for summarization (placeholder — actual
          LLM summarization requires a client reference).

        Args:
            ctx: The pipeline context to update.
            fidelity: The resolved fidelity mode string.
        """
        ctx.set("_fidelity_mode", fidelity)
        logger.debug("Fidelity mode set to '%s'", fidelity)

        if fidelity == "full":
            # Preserve everything
            return

        if fidelity == "truncate":
            data = ctx.to_dict()
            for key, value in data.items():
                if key.startswith("_"):
                    continue  # Skip internal keys
                if isinstance(value, str) and len(value) > _TRUNCATE_MAX_VALUE_LEN:
                    ctx.set(
                        key,
                        value[:_TRUNCATE_MAX_VALUE_LEN] + "... [truncated]",
                    )
            return

        if fidelity == "compact":
            data = ctx.to_dict()
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, str) and len(value) > _TRUNCATE_MAX_VALUE_LEN:
                    ctx.set(key, f"[compact: {len(value)} chars omitted]")
                elif isinstance(value, (list, dict)):
                    serialized = str(value)
                    if len(serialized) > _TRUNCATE_MAX_VALUE_LEN:
                        ctx.set(key, f"[compact: {type(value).__name__} omitted]")
            return

        if fidelity.startswith("summary:"):
            # Mark context for summarization — actual LLM call is
            # outside the engine's scope.  Store the detail level so
            # downstream consumers can act on it.
            detail = fidelity.split(":", 1)[1]
            ctx.set("_summary_detail", detail)
            ctx.set("_needs_summarization", True)
            return

    @staticmethod
    def _find_incoming_edge(
        node_name: str,
        pipeline: Pipeline,
        prev_node: str | None,
    ) -> PipelineEdge | None:
        """Find the incoming edge from *prev_node* to *node_name*.

        Args:
            node_name: The target node.
            pipeline: The pipeline definition.
            prev_node: The previous node, or ``None`` for the start.

        Returns:
            The matching edge, or ``None``.
        """
        if prev_node is None:
            return None
        for edge in pipeline.edges:
            if edge.source == prev_node and edge.target == node_name:
                return edge
        return None

    @staticmethod
    def _check_goal_gates(
        goal_gate_nodes: list[PipelineNode],
        node_outcomes: dict[str, OutcomeStatus],
    ) -> PipelineNode | None:
        """Return the first unsatisfied goal gate node, or ``None``."""
        for gn in goal_gate_nodes:
            outcome = node_outcomes.get(gn.name)
            if outcome not in (
                OutcomeStatus.SUCCESS,
                OutcomeStatus.PARTIAL_SUCCESS,
            ):
                return gn
        return None

    @staticmethod
    def _find_retry_target(
        node: PipelineNode, pipeline: Pipeline
    ) -> str | None:
        """Find retry target per spec §3.4.

        Resolution order: node.retry_target → node.fallback_retry_target
        → pipeline.retry_target → pipeline.fallback_retry_target.
        """
        if node.retry_target and node.retry_target in pipeline.nodes:
            return node.retry_target
        if (
            node.fallback_retry_target
            and node.fallback_retry_target in pipeline.nodes
        ):
            return node.fallback_retry_target
        if pipeline.retry_target and pipeline.retry_target in pipeline.nodes:
            return pipeline.retry_target
        if (
            pipeline.fallback_retry_target
            and pipeline.fallback_retry_target in pipeline.nodes
        ):
            return pipeline.fallback_retry_target
        return None

    @staticmethod
    def _find_edge(
        source: str, target: str, pipeline: Pipeline
    ) -> PipelineEdge | None:
        """Find the edge from *source* to *target*."""
        for edge in pipeline.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None

    def _resolve_next(
        self,
        result: NodeResult,
        node: PipelineNode,
        pipeline: Pipeline,
        ctx: PipelineContext,
    ) -> str | None:
        """5-step edge selection per spec §3.3."""
        # Explicit routing override from handler
        if result.next_node:
            return result.next_node

        # Terminal node — stop
        if node.is_terminal:
            return None

        edges = pipeline.outgoing_edges(node.name)
        if not edges:
            return None  # Implicitly terminal

        extra_vars = {
            "outcome": result.status.value,
            "preferred_label": result.preferred_label or "",
        }

        # Step 1: Condition matching — short-circuit if any match (E1)
        condition_matched: list[PipelineEdge] = []
        for edge in edges:
            if edge.condition is not None:
                try:
                    if evaluate_condition(
                        edge.condition, ctx, extra_vars=extra_vars
                    ):
                        condition_matched.append(edge)
                except Exception as exc:
                    logger.error(
                        "Error evaluating condition '%s': %s",
                        edge.condition,
                        exc,
                    )
                    ctx.set("_condition_error", str(exc))

        # E1: If ANY condition matched, return best immediately —
        # do NOT mix with unconditional edges
        if condition_matched:
            return self._best_by_weight_then_lexical(condition_matched).target

        # Step 2: Preferred label match (E2: strip accelerator prefixes)
        if result.preferred_label:
            normalized_pref = self._normalize_label(result.preferred_label)
            for edge in edges:
                if self._normalize_label(edge.label) == normalized_pref:
                    return edge.target

        # Step 3: Suggested next IDs
        if result.suggested_next_ids:
            for suggested_id in result.suggested_next_ids:
                for edge in edges:
                    if edge.target == suggested_id:
                        return edge.target

        # E15: Steps 4/5 — filter to unconditional-only edges
        unconditional = [e for e in edges if e.condition is None]
        if unconditional:
            return self._best_by_weight_then_lexical(unconditional).target

        # E16: Final fallback — try ANY edge before raising error
        return self._best_by_weight_then_lexical(edges).target

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize a label for comparison per spec §3.3.

        Lowercase, trim whitespace, strip accelerator prefixes
        (``[Y] ``, ``Y) ``, ``Y - ``).
        """
        s = label.strip().lower()
        # Strip accelerator prefixes: [X] ..., X) ..., X - ...
        s = re.sub(r"^\[.\]\s*", "", s)
        s = re.sub(r"^.\)\s*", "", s)
        s = re.sub(r"^.\s*-\s*", "", s)
        return s

    @staticmethod
    def _best_by_weight_then_lexical(
        edges: list[PipelineEdge],
    ) -> PipelineEdge:
        """Select best edge: highest weight, then lexically first target."""
        return sorted(edges, key=lambda e: (-e.weight, e.target))[0]

    def _retry_delay(self, attempt: int, node: PipelineNode) -> float:
        """Calculate retry delay using RetryPolicy or default backoff.

        Resolution order: node.retry_policy → engine default_retry_policy
        → hardcoded exponential backoff (200ms initial, 2x factor, 60s max).
        """
        policy = node.retry_policy or self._default_retry_policy
        if policy:
            return policy.backoff.calculate_delay(attempt)
        # Legacy fallback — matches default BackoffConfig values
        base = 0.2  # 200ms initial
        factor = 2.0
        delay = base * (factor ** (attempt - 1))
        # E6: max_delay is 60s (not 30s)
        max_delay = 60.0
        delay = min(delay, max_delay)
        # E6: Jitter range [0.5x, 1.5x]
        return max(0, delay * random.uniform(0.5, 1.5))

    def _save_checkpoint(
        self,
        pipeline_name: str,
        current_node: str,
        ctx: PipelineContext,
        completed: list[str],
    ) -> None:
        """Save checkpoint with node_retries extracted from context."""
        if self._checkpoint_dir is None:
            return

        # S3-ckpt: Extract node_retries from context keys
        node_retries: dict[str, int] = {}
        for key, value in ctx.to_dict().items():
            if key.startswith("internal.retry_count."):
                node_name = key[len("internal.retry_count."):]
                if isinstance(value, int):
                    node_retries[node_name] = value

        cp = Checkpoint(
            pipeline_name=pipeline_name,
            current_node=current_node,
            context=ctx,
            completed_nodes=list(completed),
            timestamp=time.time(),
            node_retries=node_retries,
        )
        path = save_checkpoint(cp, self._checkpoint_dir)
        logger.debug("Checkpoint saved to %s", path)

    async def _save_checkpoint_and_emit(
        self,
        pipeline_name: str,
        current_node: str,
        ctx: PipelineContext,
        completed: list[str],
    ) -> None:
        """Save checkpoint and emit a CHECKPOINT_SAVED event."""
        self._save_checkpoint(pipeline_name, current_node, ctx, completed)
        if self._checkpoint_dir is not None:
            await self._emit(PipelineEvent(
                type=PipelineEventType.CHECKPOINT_SAVED,
                node_name=current_node,
                pipeline_name=pipeline_name,
                data={"node_id": current_node},
            ))
