"""Pluggable node handlers for pipeline execution.

Each handler implements the :class:`NodeHandler` protocol — a single
``execute`` async method that receives the node definition and shared
context, and returns a :class:`NodeResult`.

The :class:`HandlerRegistry` maps handler-type strings to handler
instances and is used by the engine to dispatch execution.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
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
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult: ...


class HandlerRegistry:
    """Registry mapping handler-type strings to handler instances.

    Used by :class:`PipelineEngine` to dispatch node execution to the
    appropriate handler based on the node's ``handler_type`` attribute.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, NodeHandler] = {}

    def register(self, handler_type: str, handler: NodeHandler) -> None:
        """Register a *handler* under *handler_type*.

        Args:
            handler_type: Dispatch key (e.g. ``"codergen"``).
            handler: The handler instance to register.
        """
        self._handlers[handler_type] = handler

    def get(self, handler_type: str) -> NodeHandler | None:
        """Return the handler for *handler_type*, or ``None``.

        Args:
            handler_type: The handler key to look up.

        Returns:
            The registered handler, or ``None`` if not found.
        """
        return self._handlers.get(handler_type)

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

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, notes="Pipeline started")


class ExitHandler:
    """Exit point handler — no-op, always succeeds."""

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, notes="Pipeline exiting")


class CodergenHandler:
    """Invoke the Attractor coding agent to execute a prompt.

    Attempts to import ``attractor.agent``; falls back to an error result
    when the agent module is unavailable.
    """

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Run the coding agent for *node*'s prompt.

        Builds a :class:`Session` with the appropriate provider profile,
        a local execution environment, and an LLM client, then iterates
        over the event stream to collect the final text output.

        Args:
            node: The pipeline node being executed.
            context: Shared pipeline context for variable interpolation.

        Returns:
            A :class:`NodeResult` with the agent's text output on
            success, or an error description on failure.
        """
        prompt = node.attributes.get("prompt", "")
        model = node.attributes.get("model", "")

        # Interpolate context variables in the prompt
        for key, value in context.to_dict().items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

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

            result = "".join(output_parts)
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=result,
                context_updates={"last_codergen_output": result},
            )
        except ImportError:
            logger.warning(
                "attractor.agent not available — codergen handler cannot run"
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
            return NodeResult(status=OutcomeStatus.FAIL, failure_reason=str(exc))


class WaitHumanHandler:
    """Present a prompt to a human interviewer and gate on approval."""

    def __init__(
        self,
        interviewer: Any = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self._interviewer = interviewer
        self._pipeline = pipeline

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Gate execution on human approval.

        When a pipeline is available, derives choices from outgoing edge
        labels and uses multiple-choice selection. Falls back to simple
        confirm when no pipeline is set.

        Args:
            node: The pipeline node being executed.
            context: Shared pipeline context.

        Returns:
            A :class:`NodeResult` with ``approved`` in context_updates
            and ``preferred_label`` set to the chosen edge label.
        """
        prompt = node.attributes.get("prompt", "Approve this step?")
        interviewer = self._interviewer or context.get("_interviewer")

        if interviewer is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No interviewer configured for wait.human node",
            )

        await interviewer.inform(f"Node '{node.name}': {prompt}")

        # Multiple-choice from outgoing edges when pipeline is available
        if self._pipeline is not None:
            edges = self._pipeline.outgoing_edges(node.name)
            edge_labels = [e.label or e.target for e in edges]
            if edge_labels and hasattr(interviewer, "ask"):
                answer = await interviewer.ask(prompt, options=edge_labels)
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    context_updates={"approved": True},
                    preferred_label=answer,
                )

        # Fallback: simple confirm
        approved = await interviewer.confirm(prompt)
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            context_updates={"approved": approved},
        )


# Backward-compat alias
HumanGateHandler = WaitHumanHandler


class ConditionalHandler:
    """Evaluate conditions on outgoing edges and route accordingly.

    This handler does not perform work itself — it simply evaluates
    the pipeline's outgoing edges and picks the first matching target
    via ``next_node``.  The engine uses this to decide the next step.
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self._pipeline = pipeline

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Evaluate outgoing edge conditions and route to the first match.

        Args:
            node: The conditional node whose edges are evaluated.
            context: Shared pipeline context for condition resolution.

        Returns:
            A :class:`NodeResult` with ``next_node`` set to the first
            matching edge target, or the default (unconditional) edge.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return NodeResult(status=OutcomeStatus.SUCCESS)

        edges = pipeline.outgoing_edges(node.name)
        default_target: str | None = None

        for edge in edges:
            if edge.condition is None:
                default_target = edge.target
                continue
            if evaluate_condition(edge.condition, context):
                return NodeResult(status=OutcomeStatus.SUCCESS, next_node=edge.target)

        if default_target:
            return NodeResult(status=OutcomeStatus.SUCCESS, next_node=default_target)

        return NodeResult(status=OutcomeStatus.SUCCESS)


class ParallelHandler:
    """Fan-out execution across multiple sub-paths.

    The node's ``branches`` attribute should be a list of node names.
    Each branch gets its own scoped context and runs concurrently via
    ``asyncio.gather``.  Results are merged back into the parent context.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Fan out execution across the branches listed in *node*.

        Args:
            node: The parallel node whose ``branches`` attribute lists
                the sub-nodes to execute concurrently.
            context: Shared pipeline context.

        Returns:
            A :class:`NodeResult` with merged outputs from all branches.
        """
        branches = node.attributes.get("branches", [])
        if isinstance(branches, str):
            branches = [b.strip() for b in branches.split(",")]

        if not branches:
            return NodeResult(
                status=OutcomeStatus.SUCCESS, output="No branches specified"
            )

        pipeline = self._pipeline
        registry = self._registry

        if not pipeline or not registry:
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=f"[stub] Would run parallel branches: {branches}",
            )

        async def _run_branch(branch_name: str) -> tuple[str, NodeResult]:
            branch_node = pipeline.nodes.get(branch_name)
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
            result = await handler.execute(branch_node, scope)
            context.merge_scope(scope, branch_name)
            if result.context_updates:
                for k, v in result.context_updates.items():
                    context.set(f"{branch_name}.{k}", v)
            return branch_name, result

        results = await asyncio.gather(
            *[_run_branch(b) for b in branches], return_exceptions=True
        )

        outputs: dict[str, Any] = {}
        all_success = True
        for item in results:
            if isinstance(item, BaseException):
                all_success = False
                outputs["_error"] = str(item)
            else:
                name, res = item
                outputs[name] = res.output
                if not res.success:
                    all_success = False

        return NodeResult(
            status=OutcomeStatus.SUCCESS if all_success else OutcomeStatus.FAIL,
            output=outputs,
            context_updates={"parallel.results": outputs},
        )


class FanInHandler:
    """Consolidate parallel branch results."""

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        results = context.get("parallel.results", {})
        if not results:
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output="No parallel results to consolidate",
            )

        # Simple heuristic: pick branch with best outcome
        best_id = None
        best_output = None
        for branch_id, branch_data in results.items():
            if best_id is None:
                best_id = branch_id
                best_output = branch_data

        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            output=best_output,
            context_updates={"parallel.fan_in.best_id": best_id},
        )


class ToolHandler:
    """Execute a shell command and capture exit code + output."""

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Run a shell command defined in *node*'s ``tool_command`` attribute.

        Falls back to the ``command`` attribute for backward compatibility.

        Args:
            node: The pipeline node with a ``tool_command`` (or ``command``) attribute.
            context: Shared pipeline context for variable interpolation.

        Returns:
            A :class:`NodeResult` with stdout/stderr and exit code.
        """
        command = node.attributes.get("tool_command") or node.attributes.get(
            "command", ""
        )
        if not command:
            return NodeResult(
                status=OutcomeStatus.FAIL, failure_reason="No command specified"
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

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        """Run an iterative refinement loop on a sub-pipeline.

        Args:
            node: The manager loop node with ``sub_pipeline``,
                ``done_condition``, and ``max_iterations`` attributes.
            context: Shared pipeline context.

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
    registry = HandlerRegistry()
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler())
    registry.register(
        "wait.human", WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
    )
    registry.register("conditional", ConditionalHandler(pipeline=pipeline))

    parallel = ParallelHandler(registry=registry, pipeline=pipeline)
    registry.register("parallel", parallel)
    registry.register("parallel.fan_in", FanInHandler())
    registry.register("tool", ToolHandler())
    registry.register("stack.manager_loop", ManagerLoopHandler())
    return registry
