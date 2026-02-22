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
from attractor.pipeline.events import PipelineEvent, PipelineEventEmitter, PipelineEventType
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
        event_emitter: PipelineEventEmitter | None = None,
    ) -> None:
        self._interviewer = interviewer
        self._pipeline = pipeline
        self._event_emitter = event_emitter

    async def _emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event if an emitter is configured."""
        if self._event_emitter is not None:
            await self._event_emitter.emit(event)

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
                await self._emit(PipelineEvent(
                    type=PipelineEventType.INTERVIEW_STARTED,
                    node_name=node.name,
                    data={"prompt": prompt},
                ))
                try:
                    ask_coro = interviewer.ask(question)
                    if timeout is not None:
                        raw_answer = await asyncio.wait_for(
                            ask_coro, timeout=timeout
                        )
                    else:
                        raw_answer = await ask_coro
                except asyncio.TimeoutError:
                    await self._emit(PipelineEvent(
                        type=PipelineEventType.INTERVIEW_TIMEOUT,
                        node_name=node.name,
                        data={"prompt": prompt, "timeout": timeout},
                    ))
                    return self._handle_timeout(node, edges)

                # Normalize to Answer for backward compat with mocked interviewers
                if isinstance(raw_answer, Answer):
                    answer = raw_answer
                else:
                    answer = Answer(value=str(raw_answer))

                await self._emit(PipelineEvent(
                    type=PipelineEventType.INTERVIEW_COMPLETED,
                    node_name=node.name,
                    data={"prompt": prompt, "answer": str(answer.value)},
                ))

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
        await self._emit(PipelineEvent(
            type=PipelineEventType.INTERVIEW_STARTED,
            node_name=node.name,
            data={"prompt": prompt},
        ))
        try:
            confirm_coro = interviewer.confirm(prompt)
            if timeout is not None:
                approved = await asyncio.wait_for(confirm_coro, timeout=timeout)
            else:
                approved = await confirm_coro
        except asyncio.TimeoutError:
            await self._emit(PipelineEvent(
                type=PipelineEventType.INTERVIEW_TIMEOUT,
                node_name=node.name,
                data={"prompt": prompt, "timeout": timeout},
            ))
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

        await self._emit(PipelineEvent(
            type=PipelineEventType.INTERVIEW_COMPLETED,
            node_name=node.name,
            data={"prompt": prompt, "answer": str(approved)},
        ))

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
    ``join_policy`` (wait_all/k_of_n/first_success/quorum) and
    ``error_policy`` (fail_fast/continue/ignore) attributes.

    Join policies:
        - ``wait_all``: All branches must complete. SUCCESS if none fail,
          PARTIAL_SUCCESS otherwise.
        - ``k_of_n``: At least ``k`` branches must succeed (``k`` from
          node attribute, default 1).
        - ``first_success``: SUCCESS as soon as one branch succeeds.
        - ``quorum``: Majority of branches must succeed.

    Error policies:
        - ``continue``: Run all branches, collect all results (default).
        - ``fail_fast``: Cancel remaining branches on first failure.
        - ``ignore``: Treat failures as SKIPPED; don't count them as
          failures for join evaluation.
    """

    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        pipeline: Pipeline | None = None,
        event_emitter: PipelineEventEmitter | None = None,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline
        self._event_emitter = event_emitter

    async def _emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event if an emitter is configured."""
        if self._event_emitter is not None:
            await self._event_emitter.emit(event)

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
        k_value = int(node.attributes.get("k", 1))

        await self._emit(PipelineEvent(
            type=PipelineEventType.PARALLEL_STARTED,
            node_name=node.name,
            data={"branches": branches},
        ))

        async def _run_branch(branch_name: str) -> tuple[str, NodeResult]:
            await self._emit(PipelineEvent(
                type=PipelineEventType.PARALLEL_BRANCH_STARTED,
                node_name=node.name,
                data={"branch": branch_name},
            ))
            branch_node = active_graph.nodes.get(branch_name)
            if not branch_node:
                result = NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason=f"Branch node '{branch_name}' not found",
                )
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PARALLEL_BRANCH_COMPLETED,
                    node_name=node.name,
                    data={"branch": branch_name, "status": result.status.value},
                ))
                return branch_name, result
            handler = registry.get(branch_node.handler_type)
            if not handler:
                result = NodeResult(
                    status=OutcomeStatus.FAIL,
                    failure_reason=f"No handler for '{branch_node.handler_type}'",
                )
                await self._emit(PipelineEvent(
                    type=PipelineEventType.PARALLEL_BRANCH_COMPLETED,
                    node_name=node.name,
                    data={"branch": branch_name, "status": result.status.value},
                ))
                return branch_name, result
            scope = context.create_scope(branch_name)
            result = await handler.execute(branch_node, scope, active_graph, logs_root)
            context.merge_scope(scope, branch_name)
            if result.context_updates:
                for ctx_key, v in result.context_updates.items():
                    context.set(f"{branch_name}.{ctx_key}", v)
            await self._emit(PipelineEvent(
                type=PipelineEventType.PARALLEL_BRANCH_COMPLETED,
                node_name=node.name,
                data={"branch": branch_name, "status": result.status.value},
            ))
            return branch_name, result

        # Execute based on error_policy
        if error_policy == "fail_fast":
            raw_results = await self._run_fail_fast(branches, _run_branch)
        else:
            # "continue" and "ignore" both run all branches
            raw_results = await asyncio.gather(
                *[_run_branch(b) for b in branches], return_exceptions=True
            )

        # Collect outputs
        outputs: dict[str, Any] = {}
        success_count = 0
        fail_count = 0
        for item in raw_results:
            if isinstance(item, BaseException):
                if error_policy != "ignore":
                    fail_count += 1
                outputs["_error"] = str(item)
            else:
                bname, res = item
                if res.success:
                    success_count += 1
                    outputs[bname] = res.output
                else:
                    if error_policy == "ignore":
                        # Treat failures as SKIPPED — exclude from outputs
                        pass
                    else:
                        fail_count += 1
                        outputs[bname] = res.output

        # Apply join_policy
        total = success_count + fail_count
        if join_policy == "first_success":
            status = OutcomeStatus.SUCCESS if success_count > 0 else OutcomeStatus.FAIL
        elif join_policy == "k_of_n":
            status = (
                OutcomeStatus.SUCCESS
                if success_count >= k_value
                else OutcomeStatus.FAIL
            )
        elif join_policy == "quorum":
            status = (
                OutcomeStatus.SUCCESS
                if total > 0 and success_count > total / 2
                else OutcomeStatus.FAIL
            )
        else:  # wait_all (default)
            if fail_count == 0:
                status = OutcomeStatus.SUCCESS
            else:
                status = OutcomeStatus.PARTIAL_SUCCESS

        await self._emit(PipelineEvent(
            type=PipelineEventType.PARALLEL_COMPLETED,
            node_name=node.name,
            data={
                "branches": branches,
                "success_count": success_count,
                "fail_count": fail_count,
                "status": status.value,
            },
        ))

        return NodeResult(
            status=status,
            output=outputs,
            context_updates={"parallel.results": outputs},
        )

    @staticmethod
    async def _run_fail_fast(
        branches: list[str],
        run_fn: Any,
    ) -> list[tuple[str, NodeResult] | BaseException]:
        """Run branches concurrently, cancelling remaining on first failure.

        Args:
            branches: List of branch node names.
            run_fn: Async callable taking a branch name and returning
                ``(name, NodeResult)``.

        Returns:
            List of completed results (may be fewer than total branches
            if cancellation occurred).
        """
        tasks = [asyncio.create_task(run_fn(b)) for b in branches]
        results: list[tuple[str, NodeResult] | BaseException] = []
        pending = set(tasks)

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    result = task.result()
                    results.append(result)
                    _name, res = result
                    if not res.success:
                        # Cancel all remaining
                        for p in pending:
                            p.cancel()
                        if pending:
                            await asyncio.wait(pending)
                        pending = set()
                        break
                except Exception as exc:
                    results.append(exc)
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.wait(pending)
                    pending = set()
                    break

        return results


class FanInHandler:
    """Consolidate parallel branch results.

    H12: Ranks outcomes by status priority (SUCCESS > PARTIAL > FAIL).
    P-P12: When ``node.prompt`` is set and a backend is provided, uses
    LLM-based evaluation to rank candidates.
    P-P13: Returns FAIL when results are empty or all candidates failed.
    """

    _STATUS_RANK: dict[str, int] = {
        "success": 0,
        "partial_success": 1,
        "retry": 2,
        "fail": 3,
    }

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self._backend = backend

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        """Consolidate parallel results and select the best candidate.

        Args:
            node: The fan-in node. When ``prompt`` is set and a backend
                is configured, LLM-based evaluation is used.
            context: Shared pipeline context (reads ``parallel.results``).
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` with the selected best candidate.
        """
        results = context.get("parallel.results", {})

        # P-P13: Empty results → FAIL per spec
        if not results:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No parallel results to evaluate",
            )

        # P-P13: Check if ALL candidates failed
        all_failed = True
        for branch_data in results.values():
            if isinstance(branch_data, dict) and "status" in branch_data:
                if branch_data["status"] != "fail":
                    all_failed = False
                    break
            else:
                # Non-dict data treated as success
                all_failed = False
                break

        if all_failed:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="All candidates failed",
            )

        # P-P12: LLM-based evaluation when node.prompt is set
        prompt = node.attributes.get("prompt", "") or node.prompt
        if prompt and self._backend:
            return await self._llm_evaluate(node, prompt, results, context)

        # H12: Heuristic — rank by status priority
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

    async def _llm_evaluate(
        self,
        node: PipelineNode,
        prompt: str,
        results: dict[str, Any],
        context: PipelineContext,
    ) -> NodeResult:
        """Use LLM backend to evaluate and rank candidates.

        Args:
            node: The fan-in node.
            prompt: Evaluation prompt from the node.
            results: Parallel branch results to evaluate.
            context: Shared pipeline context.

        Returns:
            A :class:`NodeResult` with the LLM-selected best candidate.
        """
        results_text = json.dumps(results, indent=2, default=str)
        full_prompt = f"{prompt}\n\nCandidate results:\n{results_text}"

        try:
            assert self._backend is not None
            evaluation = await self._backend.run(node, full_prompt, context)

            # Try to identify best_id from evaluation text
            best_id = None
            for branch_id in results:
                if branch_id in evaluation:
                    best_id = branch_id
                    break
            if best_id is None:
                best_id = next(iter(results))

            return NodeResult(
                status=OutcomeStatus.SUCCESS,
                output=evaluation,
                context_updates={
                    "parallel.fan_in.best_id": best_id,
                    "parallel.fan_in.evaluation": evaluation,
                },
                notes=f"LLM evaluation selected: {best_id}",
            )
        except Exception as exc:
            logger.exception("FanIn LLM evaluation failed")
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason=f"LLM evaluation failed: {exc}",
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
    """Orchestrates sprint-based iteration by supervising a child pipeline.

    Per spec §4.11: observes child telemetry, evaluates progress via
    stop conditions, and optionally steers the child through intervention.

    Supports two operation modes:

    1. **Child pipeline mode** (spec §4.11): Uses ``child_dotfile``,
       ``manager.poll_interval``, ``manager.max_cycles``,
       ``manager.stop_condition``, and ``manager.actions``
       (observe/steer/wait).
    2. **Legacy sub-pipeline mode**: Uses ``sub_pipeline``,
       ``max_iterations``, ``done_condition`` for backward
       compatibility.

    The mode is auto-detected: if ``child_dotfile`` is present (on the
    node or in the graph metadata), child pipeline mode is used.
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
        """Run a manager loop on a child or sub-pipeline.

        Auto-detects mode based on the presence of ``child_dotfile``.

        Args:
            node: The manager loop node.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.

        Returns:
            A :class:`NodeResult` indicating how many cycles/iterations
            ran and whether the stop/done condition was met.
        """
        child_dotfile = node.attributes.get("child_dotfile", "")
        if not child_dotfile and graph is not None:
            child_dotfile = graph.metadata.get("stack.child_dotfile", "")

        if child_dotfile:
            return await self._execute_child_pipeline(
                node, context, graph, logs_root, child_dotfile
            )
        return await self._execute_legacy(node, context, graph, logs_root)

    # ------------------------------------------------------------------
    # Child pipeline mode (spec §4.11)
    # ------------------------------------------------------------------

    async def _execute_child_pipeline(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None,
        logs_root: Path | None,
        child_dotfile: str,
    ) -> NodeResult:
        """Execute in child pipeline mode per spec §4.11.

        Args:
            node: The manager loop node.
            context: Shared pipeline context.
            graph: The full pipeline graph.
            logs_root: Filesystem path for this run's log/artifact directory.
            child_dotfile: Path to the child pipeline DOT file.

        Returns:
            A :class:`NodeResult` for the child pipeline supervision.
        """
        from attractor.pipeline.models import parse_duration

        poll_interval = parse_duration(
            node.attributes.get("manager.poll_interval", "45s")
        )
        max_cycles = int(node.attributes.get("manager.max_cycles", 1000))
        stop_condition = node.attributes.get("manager.stop_condition", "")
        actions = [
            a.strip()
            for a in node.attributes.get(
                "manager.actions", "observe,wait"
            ).split(",")
        ]

        if self._engine is None:
            return NodeResult(
                status=OutcomeStatus.FAIL,
                failure_reason="No engine configured for manager_loop handler",
            )

        # Auto-start child if configured
        autostart = node.attributes.get("stack.child_autostart", "true")
        if autostart == "true" and hasattr(self._engine, "start_child_pipeline"):
            await self._engine.start_child_pipeline(child_dotfile, context)

        for cycle in range(1, max_cycles + 1):
            context.set("_manager_cycle", cycle)

            # Observe: ingest child telemetry
            if "observe" in actions and hasattr(self._engine, "observe_child"):
                await self._engine.observe_child(context)

            # Steer: inject context into child
            if "steer" in actions and hasattr(self._engine, "steer_child"):
                await self._engine.steer_child(context, node)

            # Check child status
            child_status = context.get_string("stack.child.status", "")
            if child_status in ("completed", "failed"):
                child_outcome = context.get_string(
                    "stack.child.outcome", ""
                )
                if child_outcome == "success" or (
                    child_status == "completed" and child_outcome != "fail"
                ):
                    if logs_root is not None:
                        _write_status_file(logs_root, node, "success")
                    return NodeResult(
                        status=OutcomeStatus.SUCCESS,
                        output=(
                            f"Child pipeline completed after {cycle} cycles"
                        ),
                        context_updates={"_manager_cycles": cycle},
                        notes="Child completed",
                    )
                if child_status == "failed":
                    reason = "Child pipeline failed"
                    if logs_root is not None:
                        _write_status_file(logs_root, node, "fail", reason)
                    return NodeResult(
                        status=OutcomeStatus.FAIL,
                        failure_reason=reason,
                        context_updates={"_manager_cycles": cycle},
                    )

            # Evaluate stop condition
            if stop_condition and evaluate_condition(
                stop_condition, context
            ):
                if logs_root is not None:
                    _write_status_file(logs_root, node, "success")
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    output=(
                        f"Stop condition satisfied after {cycle} cycles"
                    ),
                    context_updates={"_manager_cycles": cycle},
                    notes="Stop condition satisfied",
                )

            # Wait: sleep for poll_interval
            if "wait" in actions:
                await asyncio.sleep(poll_interval)

        reason = f"Max cycles exceeded ({max_cycles})"
        if logs_root is not None:
            _write_status_file(logs_root, node, "fail", reason)
        return NodeResult(
            status=OutcomeStatus.FAIL,
            failure_reason=reason,
            context_updates={"_manager_cycles": max_cycles},
        )

    # ------------------------------------------------------------------
    # Legacy sub-pipeline mode (backward compat)
    # ------------------------------------------------------------------

    async def _execute_legacy(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None,
        logs_root: Path | None,
    ) -> NodeResult:
        """Legacy sub-pipeline execution mode.

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
    event_emitter: PipelineEventEmitter | None = None,
) -> HandlerRegistry:
    """Create a :class:`HandlerRegistry` pre-loaded with built-in handlers."""
    # H15: Use CodergenHandler as default handler fallback
    registry = HandlerRegistry(default_handler=CodergenHandler())
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("codergen", CodergenHandler())
    registry.register(
        "wait.human",
        WaitHumanHandler(
            interviewer=interviewer,
            pipeline=pipeline,
            event_emitter=event_emitter,
        ),
    )
    registry.register("conditional", ConditionalHandler())

    parallel = ParallelHandler(
        registry=registry, pipeline=pipeline, event_emitter=event_emitter
    )
    registry.register("parallel", parallel)
    registry.register("parallel.fan_in", FanInHandler())
    registry.register("tool", ToolHandler())
    registry.register("stack.manager_loop", ManagerLoopHandler())
    return registry
