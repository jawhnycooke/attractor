"""Pluggable node handlers for pipeline execution.

Each handler implements the :class:`NodeHandler` protocol — a single
``execute`` async method that receives the node definition, shared
context, the pipeline graph, and an optional logs root path, and
returns a :class:`NodeResult`.

The :class:`HandlerRegistry` maps handler-type strings to handler
instances and is used by the engine to dispatch execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.models import (
    NodeResult,
    OutcomeStatus,
    Pipeline,
    PipelineContext,
    PipelineNode,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class NodeHandler(Protocol):
    """Protocol that all node handlers must satisfy."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult: ...


class HandlerRegistry:
    """Registry mapping handler-type strings to handler instances.

    Used by :class:`PipelineEngine` to dispatch node execution to the
    appropriate handler based on the node's ``handler_type`` attribute.

    Supports a ``default_handler`` fallback for unknown handler types.
    """

    def __init__(self, default_handler: NodeHandler | None = None) -> None:
        self._handlers: dict[str, NodeHandler] = {}
        self._default_handler = default_handler

    @property
    def default_handler(self) -> NodeHandler | None:
        """Return the default fallback handler."""
        return self._default_handler

    @default_handler.setter
    def default_handler(self, handler: NodeHandler | None) -> None:
        """Set the default fallback handler."""
        self._default_handler = handler

    def register(self, handler_type: str, handler: NodeHandler) -> None:
        """Register a *handler* under *handler_type*.

        Args:
            handler_type: Dispatch key (e.g. ``"codergen"``).
            handler: The handler instance to register.
        """
        self._handlers[handler_type] = handler

    def get(self, handler_type: str) -> NodeHandler | None:
        """Return the handler for *handler_type*, or the default handler.

        Args:
            handler_type: The handler key to look up.

        Returns:
            The registered handler, the default handler, or ``None``.
        """
        return self._handlers.get(handler_type, self._default_handler)

    def has(self, handler_type: str) -> bool:
        """Return ``True`` if *handler_type* is registered.

        Args:
            handler_type: The handler key to check.

        Returns:
            Whether the handler type is registered.
        """
        return handler_type in self._handlers

    @property
    def registered_types(self) -> list[str]:
        """Return a list of all registered handler type strings."""
        return list(self._handlers.keys())


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


class StartHandler:
    """Entry point handler — no-op, always succeeds."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, notes="Pipeline started")


class ExitHandler:
    """Exit point handler — no-op, always succeeds."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, notes="Pipeline exiting")


def _expand_goal(prompt: str, graph: Pipeline | None) -> str:
    """Replace ``$goal`` in *prompt* with the pipeline's goal attribute."""
    if graph is not None:
        goal = graph.metadata.get("goal", graph.goal or "")
        prompt = prompt.replace("$goal", str(goal))
    return prompt


class CodergenHandler:
    """Invoke the Attractor coding agent to execute a prompt.

    Attempts to import ``attractor.agent``; falls back to an error result
    when the agent module is unavailable.
    """

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Run the coding agent for *node*'s prompt.

        Args:
            node: The pipeline node being executed.
            context: Shared pipeline context for variable interpolation.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` with the agent's text output on
            success, or an error description on failure.
        """
        prompt = node.attributes.get("prompt", "") or node.prompt
        model = node.attributes.get("model", "")

        # H3: Fall back to node.label when prompt is empty
        if not prompt:
            prompt = node.label or ""

        # H4: Expand $goal variable
        prompt = _expand_goal(prompt, graph)

        # Interpolate context variables in the prompt
        for key, value in context.to_dict().items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # H2: Write prompt.md when logs_root is provided
        stage_dir: Path | None = None
        if logs_root is not None:
            stage_dir = Path(logs_root) / node.name
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "prompt.md").write_text(prompt)

        try:
            from attractor.agent.events import AgentEventType
            from attractor.agent.environment import LocalExecutionEnvironment
            from attractor.agent.profiles import (
                AnthropicProfile,
                GeminiProfile,
                OpenAIProfile,
            )
            from attractor.agent.session import Session, SessionConfig
            from attractor.llm.client import LLMClient

            # Select profile based on model name prefix
            if model.startswith("claude"):
                profile = AnthropicProfile()
            elif model.startswith("gemini"):
                profile = GeminiProfile()
            else:
                profile = OpenAIProfile()

            environment = LocalExecutionEnvironment()
            config = SessionConfig(model_id=model)
            llm_client = LLMClient()

            session = Session(
                profile=profile,
                environment=environment,
                config=config,
                llm_client=llm_client,
            )

            # Consume the async event iterator and collect text
            output_parts: list[str] = []
            async for event in session.submit(prompt):
                if event.type == AgentEventType.ASSISTANT_TEXT_DELTA:
                    text = event.data.get("text", "")
                    if text:
                        output_parts.append(text)

            result_text = "".join(output_parts)

            # H2: Write response.md and status.json
            if stage_dir is not None:
                (stage_dir / "response.md").write_text(result_text)
                (stage_dir / "status.json").write_text(
                    json.dumps({"status": "success", "node": node.name})
                )

            # H5: Set last_response context key (not last_codergen_output)
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=result_text,
                context_updates={"last_response": result_text},
            )
        except ImportError:
            logger.warning(
                "attractor.agent not available — codergen handler cannot run"
            )
            if stage_dir is not None:
                (stage_dir / "status.json").write_text(
                    json.dumps({"status": "fail", "node": node.name,
                                "reason": "agent module not available"})
                )
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason=(
                    "attractor.agent module is not available; "
                    "cannot execute codergen handler"
                ),
            )
        except Exception as exc:
            logger.exception("Handler failed on node '%s'", node.name)
            if stage_dir is not None:
                (stage_dir / "status.json").write_text(
                    json.dumps({"status": "fail", "node": node.name,
                                "reason": str(exc)})
                )
            return NodeResult(status=OutcomeStatus.FAIL, failure_reason=str(exc))


def _parse_accelerator_key(label: str) -> str:
    """Extract accelerator key from an edge label.

    Patterns matched:
    - ``[K] Label`` -> ``K``
    - ``K) Label``  -> ``K``
    - ``K - Label`` -> ``K``
    - First character of label -> ``L``
    """
    # [K] Label
    m = re.match(r"^\[(\w)\]\s*", label)
    if m:
        return m.group(1)
    # K) Label
    m = re.match(r"^(\w)\)\s*", label)
    if m:
        return m.group(1)
    # K - Label
    m = re.match(r"^(\w)\s*-\s*", label)
    if m:
        return m.group(1)
    # First character
    if label:
        return label[0]
    return ""


class WaitHumanHandler:
    """Present a prompt to a human interviewer and gate on approval."""

    def __init__(
        self,
        interviewer: Any = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self._interviewer = interviewer
        self._pipeline = pipeline

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Gate execution on human approval.

        Uses outgoing edge targets as ``suggested_next_ids`` and sets
        ``human_choice`` / ``selected_edge_id`` context keys per spec.

        Args:
            node: The pipeline node being executed.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` with routing hints and context updates.
        """
        prompt = node.attributes.get("prompt", "") or node.label or "Approve this step?"
        interviewer = self._interviewer or context.get("_interviewer")

        if interviewer is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No interviewer configured for wait.human node",
            )

        # H6: Use outgoing edges from the graph parameter (preferred) or stored pipeline
        active_graph = graph or self._pipeline

        if active_graph is not None:
            edges = active_graph.outgoing_edges(node.name)
            if edges:
                # H7: Parse accelerator keys from edge labels
                edge_labels = [e.label or e.target for e in edges]

                await interviewer.inform(f"Node '{node.name}': {prompt}")

                if hasattr(interviewer, "ask"):
                    answer = await interviewer.ask(prompt, options=edge_labels)

                    # Find the matching edge
                    selected_edge = None
                    for e in edges:
                        label = e.label or e.target
                        if label == answer:
                            selected_edge = e
                            break
                    if selected_edge is None:
                        selected_edge = edges[0]

                    return NodeResult(
                        status=OutcomeStatus.SUCCESS,
                        suggested_next_ids=[selected_edge.target],
                        context_updates={
                            "human_choice": answer,
                            "selected_edge_id": selected_edge.target,
                        },
                    )

        await interviewer.inform(f"Node '{node.name}': {prompt}")

        # Fallback: simple confirm
        approved = await interviewer.confirm(prompt)
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            context_updates={"approved": approved},
        )


# Backward-compat alias
HumanGateHandler = WaitHumanHandler


class ConditionalHandler:
    """Pure no-op handler for conditional routing nodes.

    Per spec (§4.7), the handler returns SUCCESS immediately.
    Actual edge evaluation is handled by the engine's edge selection
    algorithm.
    """

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Return SUCCESS — routing is the engine's job.

        Args:
            node: The conditional node.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            An empty :class:`NodeResult` with SUCCESS status.
        """
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            notes=f"Conditional node evaluated: {node.name}",
        )


class ParallelHandler:
    """Fan-out execution across multiple sub-paths.

    Uses outgoing edges as branches per spec (§4.8). Supports
    ``join_policy`` (wait_all/first_success) and ``error_policy``
    (fail_fast/continue) attributes.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Fan out execution across outgoing edges.

        Args:
            node: The parallel node.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` with merged outputs from all branches.
        """
        active_graph = graph or self._pipeline

        # H11: Use outgoing edges as branches
        if active_graph is not None:
            edges = active_graph.outgoing_edges(node.name)
            branches = [e.target for e in edges]
        else:
            # Fallback to branches attribute
            branches = node.attributes.get("branches", [])
            if isinstance(branches, str):
                branches = [b.strip() for b in branches.split(",")]

        if not branches:
            return NodeResult(
                status=OutcomeStatus.SUCCESS, output="No branches specified"
            )

        registry = self._registry

        if not active_graph or not registry:
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=f"[stub] Would run parallel branches: {branches}",
            )

        join_policy = node.attributes.get("join_policy", "wait_all")
        error_policy = node.attributes.get("error_policy", "continue")

        async def _run_branch(branch_name: str) -> tuple[str, NodeResult]:
            branch_node = active_graph.nodes.get(branch_name)
            if not branch_node:
                return branch_name, NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason=f"Branch node '{branch_name}' not found",
                )
            handler = registry.get(branch_node.handler_type)
            if not handler:
                return branch_name, NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason=f"No handler for '{branch_node.handler_type}'",
                )
            scope = context.create_scope(branch_name)
            result = await handler.execute(branch_node, scope, active_graph, logs_root)
            context.merge_scope(scope, branch_name)
            if result.context_updates:
                for k, v in result.context_updates.items():
                    context.set(f"{branch_name}.{k}", v)
            return branch_name, result

        results = await asyncio.gather(
            *[_run_branch(b) for b in branches], return_exceptions=True
        )

        outputs: dict[str, Any] = {}
        success_count = 0
        fail_count = 0
        for item in results:
            if isinstance(item, BaseException):
                fail_count += 1
                outputs["_error"] = str(item)
            else:
                name, res = item
                outputs[name] = res.output
                if res.success:
                    success_count += 1
                else:
                    fail_count += 1

        # H11: Apply join_policy
        if join_policy == "first_success":
            if success_count > 0:
                status = OutcomeStatus.SUCCESS
            else:
                status = OutcomeStatus.FAIL
        else:  # wait_all (default)
            if fail_count == 0:
                status = OutcomeStatus.SUCCESS
            else:
                status = OutcomeStatus.PARTIAL_SUCCESS

        return NodeResult(
            status=status,
            output=outputs,
            context_updates={"parallel.results": outputs},
        )


class FanInHandler:
    """Consolidate parallel branch results.

    H12: Ranks outcomes by status priority (SUCCESS > PARTIAL > FAIL).
    """

    _STATUS_RANK: dict[str, int] = {
        "success": 0,
        "partial_success": 1,
        "retry": 2,
        "fail": 3,
    }

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        results = context.get("parallel.results", {})
        if not results:
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output="No parallel results to consolidate",
            )

        # H12: Rank by status priority
        best_id = None
        best_output = None
        best_rank = 999

        for branch_id, branch_data in results.items():
            # branch_data may be a dict with status or just output
            if isinstance(branch_data, dict) and "status" in branch_data:
                rank = self._STATUS_RANK.get(branch_data["status"], 999)
            else:
                # Treat non-dict as success
                rank = 0

            if rank < best_rank or best_id is None:
                best_rank = rank
                best_id = branch_id
                best_output = branch_data

        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            output=best_output,
            context_updates={"parallel.fan_in.best_id": best_id},
        )


class ToolHandler:
    """Execute a shell command and capture exit code + output."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Run a shell command defined in *node*'s ``tool_command`` attribute.

        H13: Only reads ``tool_command`` — no ``command`` fallback.

        Args:
            node: The pipeline node with a ``tool_command`` attribute.
            context: Shared pipeline context for variable interpolation.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` with stdout/stderr and exit code.
        """
        # H13: Only use tool_command, no command fallback
        command = node.attributes.get("tool_command", "")
        if not command:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No tool_command specified",
            )

        # Interpolate context variables
        for key, value in context.to_dict().items():
            command = command.replace(f"{{{key}}}", str(value))

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=float(node.attributes.get("timeout", 300)),
            )
            success = proc.returncode == 0
            return NodeResult(
                status=OutcomeStatus.SUCCESS if success else OutcomeStatus.FAIL,
                output=proc.stdout,
                failure_reason=proc.stderr if not success else None,
                context_updates={
                    "exit_code": proc.returncode,
                    "tool.output": proc.stdout,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )
        except subprocess.TimeoutExpired:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason=f"Command timed out: {command}",
                context_updates={"exit_code": -1},
            )
        except Exception as exc:
            logger.exception("Handler failed on node '%s'", node.name)
            return NodeResult(
                status=OutcomeStatus.FAIL, failure_reason=str(exc)
            )


class ManagerLoopHandler:
    """Iterative refinement loop.

    Executes a sub-pipeline (identified by the ``sub_pipeline`` attribute)
    repeatedly until a ``done_condition`` evaluates to True or
    ``max_iterations`` is reached.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Run an iterative refinement loop on a sub-pipeline.

        Args:
            node: The manager loop node with ``sub_pipeline``,
                ``done_condition``, and ``max_iterations`` attributes.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` indicating how many iterations ran
            and whether the done condition was met.
        """
        max_iterations = int(node.attributes.get("max_iterations", 5))
        done_condition = node.attributes.get("done_condition", "")

        if self._engine is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No engine configured for manager_loop handler",
            )

        for i in range(max_iterations):
            context.set("_supervisor_iteration", i + 1)

            # The engine runs a sub-pipeline on the same context
            sub_pipeline_name = node.attributes.get("sub_pipeline", "")
            if not sub_pipeline_name:
                return NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason="No sub_pipeline attribute specified",
                )

            # Delegate to engine (which will call back with the sub-pipeline)
            await self._engine.run_sub_pipeline(sub_pipeline_name, context)

            if done_condition and evaluate_condition(done_condition, context):
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    output=f"Manager loop done after {i + 1} iterations",
                    context_updates={"_supervisor_iterations": i + 1},
                )

        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            output=f"Manager loop reached max iterations ({max_iterations})",
            context_updates={"_supervisor_iterations": max_iterations},
        )


# Backward-compat alias
SupervisorHandler = ManagerLoopHandler


def create_default_registry(
    pipeline: Pipeline | None = None,
    interviewer: Any = None,
) -> HandlerRegistry:
    """Create a :class:`HandlerRegistry` pre-loaded with built-in handlers."""
    # H15: Use CodergenHandler as default handler fallback
    registry = HandlerRegistry(default_handler=CodergenHandler())
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler())
    registry.register(
        "wait.human", WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
    )
    registry.register("conditional", ConditionalHandler())

    parallel = ParallelHandler(registry=registry, pipeline=pipeline)
    registry.register("parallel", parallel)
    registry.register("parallel.fan_in", FanInHandler())
    registry.register("tool", ToolHandler())
    registry.register("stack.manager_loop", ManagerLoopHandler())
    return registry
