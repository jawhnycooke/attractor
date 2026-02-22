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
from attractor.pipeline.interviewer import (
    Answer,
    AnswerValue,
    Option,
    Question,
    QuestionType,
)
from attractor.pipeline.models import (
    NodeResult,
    OutcomeStatus,
    Pipeline,
    PipelineContext,
    PipelineEdge,
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


@runtime_checkable
class HandlerHook(Protocol):
    """Hook protocol for pre/post handler execution.

    Hooks are called by :meth:`HandlerRegistry.dispatch` around each
    handler execution, enabling logging, auditing, and tool-call
    interception per spec §9.7.
    """

    async def before_execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> None: ...

    async def after_execute(
        self, node: PipelineNode, context: PipelineContext, result: NodeResult
    ) -> None: ...


class HandlerRegistry:
    """Registry mapping handler-type strings to handler instances.

    Used by :class:`PipelineEngine` to dispatch node execution to the
    appropriate handler based on the node's ``handler_type`` attribute.

    Supports a ``default_handler`` fallback for unknown handler types
    and optional :class:`HandlerHook` instances called around execution.
    """

    def __init__(
        self,
        default_handler: NodeHandler | None = None,
        hooks: list[HandlerHook] | None = None,
    ) -> None:
        self._handlers: dict[str, NodeHandler] = {}
        self._default_handler = default_handler
        self._hooks: list[HandlerHook] = list(hooks) if hooks else []

    @property
    def default_handler(self) -> NodeHandler | None:
        """Return the default fallback handler."""
        return self._default_handler

    @default_handler.setter
    def default_handler(self, handler: NodeHandler | None) -> None:
        """Set the default fallback handler."""
        self._default_handler = handler

    @property
    def hooks(self) -> list[HandlerHook]:
        """Return the list of registered hooks."""
        return list(self._hooks)

    def add_hook(self, hook: HandlerHook) -> None:
        """Append a hook to the registry.

        Args:
            hook: The hook instance to add.
        """
        self._hooks.append(hook)

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

    async def dispatch(
        self,
        handler_type: str,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Look up and execute a handler with hook invocation.

        Runs all registered :class:`HandlerHook` ``before_execute``
        callbacks before the handler and ``after_execute`` callbacks
        after.

        Args:
            handler_type: The handler key to dispatch.
            node: The pipeline node being executed.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            The :class:`NodeResult` from the handler execution.
        """
        handler = self.get(handler_type)
        if handler is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason=f"No handler for '{handler_type}'",
            )

        for hook in self._hooks:
            await hook.before_execute(node, context)

        result = await handler.execute(node, context, graph, logs_root)

        for hook in self._hooks:
            await hook.after_execute(node, context, result)

        return result


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


def _write_status_file(
    logs_root: Path,
    node: PipelineNode,
    status: str,
    reason: str | None = None,
) -> None:
    """Write a standardized ``status.json`` to the node's stage directory.

    Args:
        logs_root: Root log directory for the current run.
        node: The pipeline node being executed.
        status: Outcome status string (``"success"`` or ``"fail"``).
        reason: Optional failure reason.
    """
    stage_dir = logs_root / node.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {
        "status": status,
        "node": node.name,
        "handler": node.handler_type,
    }
    if reason:
        data["reason"] = reason
    (stage_dir / "status.json").write_text(json.dumps(data))


@runtime_checkable
class CodergenBackend(Protocol):
    """Protocol for pluggable codergen execution backends.

    Implementations handle the actual LLM-driven code generation,
    decoupling the handler from any specific agent implementation.
    """

    async def run(
        self,
        node: PipelineNode,
        prompt: str,
        context: PipelineContext,
    ) -> str:
        """Execute a code generation prompt and return the result text.

        Args:
            node: The pipeline node being executed.
            prompt: The fully interpolated prompt string.
            context: Shared pipeline context.

        Returns:
            The generated text output.

        Raises:
            Exception: On any execution failure.
        """
        ...


class AgentBackend:
    """Default :class:`CodergenBackend` that uses ``attractor.agent.Session``."""

    async def run(
        self,
        node: PipelineNode,
        prompt: str,
        context: PipelineContext,
    ) -> str:
        """Run the coding agent session for the given prompt.

        Args:
            node: The pipeline node being executed.
            prompt: The fully interpolated prompt string.
            context: Shared pipeline context.

        Returns:
            The concatenated text output from the agent session.
        """
        from attractor.agent.events import AgentEventType
        from attractor.agent.environment import LocalExecutionEnvironment
        from attractor.agent.profiles import (
            AnthropicProfile,
            GeminiProfile,
            OpenAIProfile,
        )
        from attractor.agent.session import Session, SessionConfig
        from attractor.llm.client import LLMClient

        model = node.attributes.get("model", "")

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

        output_parts: list[str] = []
        async for event in session.submit(prompt):
            if event.type == AgentEventType.ASSISTANT_TEXT_DELTA:
                text = event.data.get("text", "")
                if text:
                    output_parts.append(text)

        return "".join(output_parts)


class CodergenHandler:
    """Invoke the Attractor coding agent to execute a prompt.

    Accepts an optional :class:`CodergenBackend` for pluggable execution.
    Falls back to :class:`AgentBackend` or simulation mode when the agent
    module is unavailable.
    """

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self._backend = backend

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

        # Resolve backend
        backend = self._backend
        if backend is None:
            # P-C02: Simulation mode per spec §4.5 — no backend provided
            result_text = f"[Simulated] Response for stage: {node.name}"
            if stage_dir is not None:
                (stage_dir / "response.md").write_text(result_text)
                (stage_dir / "status.json").write_text(
                    json.dumps({"status": "success", "node": node.name,
                                "notes": "simulation mode"})
                )
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=result_text,
                context_updates={"last_response": result_text},
                notes="simulation mode",
            )

        try:
            result_text = await backend.run(node, prompt, context)

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

        H7: Parses accelerator keys from edge labels and passes them
        to the interviewer alongside the labels.

        H8: Wraps interviewer calls with ``asyncio.wait_for`` when
        ``node.timeout`` is set.  On timeout, checks
        ``human.default_choice``; on SKIPPED, returns FAIL.

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

        timeout = node.timeout

        # H6: Use outgoing edges from the graph parameter (preferred) or stored pipeline
        active_graph = graph or self._pipeline

        if active_graph is not None:
            edges = active_graph.outgoing_edges(node.name)
            if edges:
                # H7: Parse accelerator keys from edge labels
                edge_labels = [e.label or e.target for e in edges]
                accelerator_keys = [
                    _parse_accelerator_key(label) for label in edge_labels
                ]

                # P-C10: Construct Question with MULTIPLE_CHOICE per spec §6.2
                options = [
                    Option(key=accel, label=label)
                    for accel, label in zip(accelerator_keys, edge_labels)
                ]
                question = Question(
                    text=prompt,
                    type=QuestionType.MULTIPLE_CHOICE,
                    options=options,
                    stage=node.name,
                    timeout_seconds=timeout,
                )

                await interviewer.inform(f"Node '{node.name}': {prompt}")

                # P-C10: Call interviewer.ask(question) per spec §6.1
                try:
                    ask_coro = interviewer.ask(question)
                    if timeout is not None:
                        raw_answer = await asyncio.wait_for(
                            ask_coro, timeout=timeout
                        )
                    else:
                        raw_answer = await ask_coro
                except asyncio.TimeoutError:
                    return self._handle_timeout(node, edges)

                # Normalize to Answer for backward compat with mocked interviewers
                if isinstance(raw_answer, Answer):
                    answer = raw_answer
                else:
                    answer = Answer(value=str(raw_answer))

                # H8: Handle SKIPPED
                if answer.value in (AnswerValue.SKIPPED, "SKIPPED"):
                    return NodeResult(
                        status=OutcomeStatus.FAIL,
                        failure_reason="human skipped interaction",
                    )

                # Find the matching edge via selected_option or value
                selected_edge = None
                if answer.selected_option is not None:
                    for e in edges:
                        label = e.label or e.target
                        if label == answer.selected_option.label:
                            selected_edge = e
                            break
                if selected_edge is None:
                    answer_val = str(answer.value)
                    for e in edges:
                        label = e.label or e.target
                        if label == answer_val:
                            selected_edge = e
                            break
                if selected_edge is None:
                    selected_edge = edges[0]

                choice_value = (
                    answer.selected_option.label
                    if answer.selected_option
                    else str(answer.value)
                )
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    suggested_next_ids=[selected_edge.target],
                    context_updates={
                        "human_choice": choice_value,
                        "selected_edge_id": selected_edge.target,
                    },
                )

        await interviewer.inform(f"Node '{node.name}': {prompt}")

        # Fallback: simple confirm
        # H8: Wrap with timeout if configured
        try:
            confirm_coro = interviewer.confirm(prompt)
            if timeout is not None:
                approved = await asyncio.wait_for(confirm_coro, timeout=timeout)
            else:
                approved = await confirm_coro
        except asyncio.TimeoutError:
            default_choice = node.attributes.get("human.default_choice")
            if default_choice:
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    context_updates={
                        "approved": True,
                        "human_choice": default_choice,
                    },
                )
            return NodeResult(
                status=OutcomeStatus.RETRY,
                failure_reason="human gate timeout, no default",
            )

        # H8: Handle SKIPPED
        if approved == "SKIPPED":
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="human skipped interaction",
            )

        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            context_updates={"approved": approved},
        )

    @staticmethod
    def _handle_timeout(
        node: PipelineNode,
        edges: list[PipelineEdge],
    ) -> NodeResult:
        """Handle timeout for the ask path.

        Args:
            node: The pipeline node being executed.
            edges: Outgoing edges from the node.

        Returns:
            A :class:`NodeResult` using the default choice or RETRY.
        """
        default_choice = node.attributes.get("human.default_choice")
        if default_choice:
            for e in edges:
                if e.target == default_choice:
                    return NodeResult(
                        status=OutcomeStatus.SUCCESS,
                        suggested_next_ids=[e.target],
                        context_updates={
                            "human_choice": e.label or e.target,
                            "selected_edge_id": e.target,
                        },
                    )
            # default_choice didn't match any edge target, use it directly
            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                suggested_next_ids=[default_choice],
                context_updates={
                    "human_choice": default_choice,
                    "selected_edge_id": default_choice,
                },
            )
        return NodeResult(
            status=OutcomeStatus.RETRY,
            failure_reason="human gate timeout, no default",
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
            result = NodeResult(
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
            # #12: Write status file
            if logs_root is not None:
                _write_status_file(
                    logs_root,
                    node,
                    "success" if success else "fail",
                    proc.stderr if not success else None,
                )
            return result
        except subprocess.TimeoutExpired:
            reason = f"Command timed out: {command}"
            if logs_root is not None:
                _write_status_file(logs_root, node, "fail", reason)
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason=reason,
                context_updates={"exit_code": -1},
            )
        except Exception as exc:
            logger.exception("Handler failed on node '%s'", node.name)
            reason = str(exc)
            if logs_root is not None:
                _write_status_file(logs_root, node, "fail", reason)
            return NodeResult(
                status=OutcomeStatus.FAIL, failure_reason=reason
            )


class ManagerLoopHandler:
    """Iterative refinement loop with supervisor pattern.

    Executes a sub-pipeline (identified by the ``sub_pipeline`` attribute)
    repeatedly until a ``done_condition`` evaluates to True,
    ``max_iterations`` is reached, or the sub-pipeline fails repeatedly
    (H14: ``max_consecutive_failures``).

    After each iteration the handler evaluates the sub-pipeline outcome,
    sets ``_supervisor_feedback`` and ``_supervisor_assessment`` context
    keys for feedback between iterations, and writes ``status.json``
    when ``logs_root`` is provided (#12).
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
                ``done_condition``, ``max_iterations``, and optional
                ``max_consecutive_failures`` attributes.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` indicating how many iterations ran
            and whether the done condition was met.
        """
        max_iterations = int(node.attributes.get("max_iterations", 5))
        done_condition = node.attributes.get("done_condition", "")
        max_consecutive_failures = int(
            node.attributes.get("max_consecutive_failures", 3)
        )

        if self._engine is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No engine configured for manager_loop handler",
            )

        consecutive_failures = 0

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

            # H14: Evaluate sub-pipeline results after each iteration
            sub_status = context.get("_sub_pipeline_status", "")
            sub_outcome = context.get("_sub_pipeline_outcome", "")

            if sub_status == "failed" or sub_outcome == "fail":
                consecutive_failures += 1
                context.set(
                    "_supervisor_assessment",
                    f"iteration {i + 1} failed",
                )
                context.set(
                    "_supervisor_feedback",
                    f"Sub-pipeline failed on iteration {i + 1}",
                )
            else:
                consecutive_failures = 0
                context.set(
                    "_supervisor_assessment",
                    f"iteration {i + 1} succeeded",
                )
                context.set("_supervisor_feedback", "")

            # H14: Early termination on repeated failures
            if consecutive_failures >= max_consecutive_failures:
                reason = (
                    f"Sub-pipeline failed {consecutive_failures} "
                    f"consecutive times"
                )
                if logs_root is not None:
                    _write_status_file(logs_root, node, "fail", reason)
                return NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason=reason,
                    context_updates={"_supervisor_iterations": i + 1},
                )

            if done_condition and evaluate_condition(done_condition, context):
                if logs_root is not None:
                    _write_status_file(logs_root, node, "success")
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    output=f"Manager loop done after {i + 1} iterations",
                    context_updates={"_supervisor_iterations": i + 1},
                )

        if logs_root is not None:
            _write_status_file(logs_root, node, "success")
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
