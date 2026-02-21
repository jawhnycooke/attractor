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
    """Maps handler-type strings to :class:`NodeHandler` instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, NodeHandler] = {}

    def register(self, handler_type: str, handler: NodeHandler) -> None:
        self._handlers[handler_type] = handler

    def get(self, handler_type: str) -> NodeHandler | None:
        return self._handlers.get(handler_type)

    def has(self, handler_type: str) -> bool:
        return handler_type in self._handlers

    @property
    def registered_types(self) -> list[str]:
        return list(self._handlers.keys())


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


class CodergenHandler:
    """Invoke the Attractor coding agent to execute a prompt.

    Attempts to import ``attractor.agent``; falls back to a stub
    that echoes the prompt when the agent module is unavailable.
    """

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        prompt = node.attributes.get("prompt", "")
        model = node.attributes.get("model", "")

        # Interpolate context variables in the prompt
        for key, value in context.data.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        try:
            from attractor.agent import Session  # type: ignore[import-untyped]

            session = Session(model=model) if model else Session()
            result = await session.submit(prompt)
            return NodeResult(
                success=True,
                output=result,
                context_updates={"last_codergen_output": result},
            )
        except ImportError:
            logger.warning(
                "attractor.agent not available — using codergen stub"
            )
            return NodeResult(
                success=True,
                output=f"[stub] Would execute prompt: {prompt}",
                context_updates={"last_codergen_output": f"[stub] {prompt}"},
            )
        except Exception as exc:
            return NodeResult(success=False, error=str(exc))


class HumanGateHandler:
    """Present a prompt to a human interviewer and gate on approval."""

    def __init__(self, interviewer: Any = None) -> None:
        self._interviewer = interviewer

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        prompt = node.attributes.get("prompt", "Approve this step?")
        interviewer = self._interviewer or context.get("_interviewer")

        if interviewer is None:
            logger.warning("No interviewer configured — auto-approving")
            return NodeResult(
                success=True,
                context_updates={"approved": True},
            )

        await interviewer.inform(
            f"Node '{node.name}': {prompt}"
        )
        approved = await interviewer.confirm(prompt)
        return NodeResult(
            success=True,
            context_updates={"approved": approved},
        )


class ConditionalHandler:
    """Evaluate conditions on outgoing edges and route accordingly.

    This handler does not perform work itself — it simply evaluates
    the pipeline's outgoing edges and picks the first matching target
    via ``next_node``.  The engine uses this to decide the next step.
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self._pipeline = pipeline

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        pipeline = self._pipeline
        if pipeline is None:
            return NodeResult(success=True)

        edges = pipeline.outgoing_edges(node.name)
        default_target: str | None = None

        for edge in edges:
            if edge.condition is None:
                default_target = edge.target
                continue
            if evaluate_condition(edge.condition, context):
                return NodeResult(success=True, next_node=edge.target)

        if default_target:
            return NodeResult(success=True, next_node=default_target)

        return NodeResult(success=True)


class ParallelHandler:
    """Fan-out execution across multiple sub-paths.

    The node's ``branches`` attribute should be a list of node names.
    Each branch gets its own scoped context and runs concurrently via
    ``asyncio.gather``.  Results are merged back into the parent context.
    """

    def __init__(self, registry: HandlerRegistry | None = None, pipeline: Pipeline | None = None) -> None:
        self._registry = registry
        self._pipeline = pipeline

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        branches = node.attributes.get("branches", [])
        if isinstance(branches, str):
            branches = [b.strip() for b in branches.split(",")]

        if not branches:
            return NodeResult(success=True, output="No branches specified")

        pipeline = self._pipeline
        registry = self._registry

        if not pipeline or not registry:
            return NodeResult(
                success=True,
                output=f"[stub] Would run parallel branches: {branches}",
            )

        async def _run_branch(branch_name: str) -> tuple[str, NodeResult]:
            branch_node = pipeline.nodes.get(branch_name)
            if not branch_node:
                return branch_name, NodeResult(
                    success=False, error=f"Branch node '{branch_name}' not found"
                )
            handler = registry.get(branch_node.handler_type)
            if not handler:
                return branch_name, NodeResult(
                    success=False,
                    error=f"No handler for '{branch_node.handler_type}'",
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
            success=all_success,
            output=outputs,
            context_updates={"parallel_results": outputs},
        )


class ToolHandler:
    """Execute a shell command and capture exit code + output."""

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        command = node.attributes.get("command", "")
        if not command:
            return NodeResult(success=False, error="No command specified")

        # Interpolate context variables
        for key, value in context.data.items():
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
                success=success,
                output=proc.stdout,
                error=proc.stderr if not success else None,
                context_updates={
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )
        except subprocess.TimeoutExpired:
            return NodeResult(
                success=False,
                error=f"Command timed out: {command}",
                context_updates={"exit_code": -1},
            )
        except Exception as exc:
            return NodeResult(success=False, error=str(exc))


class SupervisorHandler:
    """Iterative refinement loop.

    Executes a sub-pipeline (identified by the ``sub_pipeline`` attribute)
    repeatedly until a ``done_condition`` evaluates to True or
    ``max_iterations`` is reached.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult:
        max_iterations = int(node.attributes.get("max_iterations", 5))
        done_condition = node.attributes.get("done_condition", "")

        if self._engine is None:
            return NodeResult(
                success=True,
                output=f"[stub] Would run supervisor loop up to {max_iterations}x",
            )

        for i in range(max_iterations):
            context.set("_supervisor_iteration", i + 1)

            # The engine runs a sub-pipeline on the same context
            sub_pipeline_name = node.attributes.get("sub_pipeline", "")
            if not sub_pipeline_name:
                return NodeResult(
                    success=False, error="No sub_pipeline attribute specified"
                )

            # Delegate to engine (which will call back with the sub-pipeline)
            # For now, this is a hook point — the engine injects itself.
            await self._engine.run_sub_pipeline(sub_pipeline_name, context)

            if done_condition and evaluate_condition(done_condition, context):
                return NodeResult(
                    success=True,
                    output=f"Supervisor done after {i + 1} iterations",
                    context_updates={"_supervisor_iterations": i + 1},
                )

        return NodeResult(
            success=True,
            output=f"Supervisor reached max iterations ({max_iterations})",
            context_updates={"_supervisor_iterations": max_iterations},
        )


def create_default_registry(
    pipeline: Pipeline | None = None,
    interviewer: Any = None,
) -> HandlerRegistry:
    """Create a :class:`HandlerRegistry` pre-loaded with built-in handlers."""
    registry = HandlerRegistry()
    registry.register("codergen", CodergenHandler())
    registry.register("human_gate", HumanGateHandler(interviewer=interviewer))
    registry.register("conditional", ConditionalHandler(pipeline=pipeline))

    parallel = ParallelHandler(registry=registry, pipeline=pipeline)
    registry.register("parallel", parallel)
    registry.register("tool", ToolHandler())
    registry.register("supervisor", SupervisorHandler())
    return registry
