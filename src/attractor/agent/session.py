"""Agent session — the top-level API for driving the coding agent.

A Session wraps an AgentLoop, manages conversation history, and exposes
async-iterator-based event delivery.
"""

from __future__ import annotations

import asyncio
import enum
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.events import AgentEvent, AgentEventType, EventEmitter
from attractor.agent.loop import AgentLoop, LLMClientProtocol, LoopConfig
from attractor.agent.loop_detection import LoopDetector
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.core_tools import (
    EDIT_FILE_DEF,
    GLOB_DEF,
    GREP_DEF,
    LIST_DIR_DEF,
    READ_FILE_DEF,
    SHELL_DEF,
    WRITE_FILE_DEF,
    edit_file,
    glob_tool,
    grep_tool,
    list_dir_tool,
    read_file,
    shell,
    write_file,
)
from attractor.agent.tools.apply_patch import APPLY_PATCH_DEF, apply_patch
from attractor.agent.tools.registry import ToolRegistry, ToolResult
from attractor.agent.tools.subagent import (
    CLOSE_AGENT_DEF,
    SEND_INPUT_DEF,
    SPAWN_AGENT_DEF,
    WAIT_DEF,
    close_agent,
    send_input,
    spawn_agent,
    wait_agent,
)
from attractor.agent.truncation import TruncationConfig
from attractor.agent.turns import (
    AssistantTurn,
    SteeringTurn,
    ToolResultsTurn,
    Turn,
    UserTurn,
)
from attractor.llm.models import Message, ReasoningEffort

# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration for an agent session.

    Attributes:
        max_turns: Maximum total LLM round-trips across the entire
            session. 0 = unlimited.
        max_tool_rounds_per_input: Maximum tool execution rounds within
            a single submit(). 0 = unlimited.
        default_command_timeout_ms: Default timeout for shell commands.
        max_command_timeout_ms: Upper bound for shell command timeouts.
            Even if the model requests a longer timeout, this ceiling
            is enforced.
        reasoning_effort: Optional reasoning effort level for the LLM.
        enable_loop_detection: Whether to detect repeating tool patterns.
        loop_detection_window: Number of recent tool calls to consider
            when checking for repeating patterns.
        max_subagent_depth: Max nesting level for subagents.
        model_id: Model identifier string.
        user_instructions: Extra user instructions for the system prompt.
        truncation_config: Optional truncation settings for tool output.
        context_window_warning_threshold: Fraction (0.0-1.0) of context
            window usage that triggers a warning event.
    """

    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 0
    default_command_timeout_ms: int | None = None
    max_command_timeout_ms: int = 600_000
    reasoning_effort: ReasoningEffort | None = None
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1
    model_id: str = ""
    user_instructions: str = ""
    truncation_config: TruncationConfig | None = None
    context_window_warning_threshold: float = 0.8


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionState(str, enum.Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


class Session:
    """Top-level session managing conversation history and agent loop.

    Usage::

        session = Session(profile, environment, config, llm_client)
        async for event in session.submit("Fix the bug in auth.py"):
            print(event.type, event.data)
    """

    def __init__(
        self,
        profile: ProviderProfile,
        environment: ExecutionEnvironment,
        config: SessionConfig,
        llm_client: LLMClientProtocol,
        *,
        _depth: int = 0,
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self._profile = profile
        self._env = environment
        self._config = config
        self._llm = llm_client
        self._state = SessionState.IDLE
        self._history: list[Message] = []
        self._turns: list[Turn] = []
        self._follow_up_queue: list[str] = []
        self._reasoning_effort = config.reasoning_effort
        self._registry = self._build_registry()
        self._loop_detector = LoopDetector()
        self._current_loop: AgentLoop | None = None
        self._abort_event = asyncio.Event()
        self._depth = _depth

        # Install session factory on the environment for subagent spawning
        if _depth < config.max_subagent_depth:
            self._env._session_factory = self._create_child_session  # type: ignore[attr-defined]

    def _create_child_session(
        self,
        model_override: str | None = None,
        max_turns_override: int = 0,
    ) -> "Session":
        """Create a child Session for subagent use."""
        child_config = SessionConfig(
            max_turns=max_turns_override or self._config.max_turns,
            max_tool_rounds_per_input=self._config.max_tool_rounds_per_input,
            default_command_timeout_ms=self._config.default_command_timeout_ms,
            max_command_timeout_ms=self._config.max_command_timeout_ms,
            reasoning_effort=self._config.reasoning_effort,
            enable_loop_detection=self._config.enable_loop_detection,
            loop_detection_window=self._config.loop_detection_window,
            max_subagent_depth=self._config.max_subagent_depth,
            model_id=model_override or self._config.model_id,
            user_instructions=self._config.user_instructions,
            truncation_config=self._config.truncation_config,
            context_window_warning_threshold=self._config.context_window_warning_threshold,
        )
        return Session(
            profile=self._profile,
            environment=self._env,
            config=child_config,
            llm_client=self._llm,
            _depth=self._depth + 1,
        )

    def _event(
        self,
        event_type: AgentEventType,
        data: dict[str, object] | None = None,
    ) -> AgentEvent:
        """Create an AgentEvent tagged with this session's ID."""
        return AgentEvent(
            type=event_type,
            data=data or {},
            session_id=self.id,
        )

    @property
    def state(self) -> SessionState:
        """Current session lifecycle state."""
        return self._state

    @property
    def conversation_history(self) -> list[Message]:
        """Snapshot of the conversation history (defensive copy)."""
        return list(self._history)

    @property
    def turns(self) -> list[Turn]:
        """Snapshot of the typed turn history (defensive copy)."""
        return list(self._turns)

    def _build_registry(self) -> ToolRegistry:
        """Wire up tool handlers based on the active profile."""
        registry = ToolRegistry()
        tool_names = {td.name for td in self._profile.tool_definitions}

        # Map names to (handler, definition) pairs
        _tool_map = {
            "read_file": (read_file, READ_FILE_DEF),
            "write_file": (write_file, WRITE_FILE_DEF),
            "edit_file": (edit_file, EDIT_FILE_DEF),
            "shell": (shell, SHELL_DEF),
            "grep": (grep_tool, GREP_DEF),
            "glob": (glob_tool, GLOB_DEF),
            "list_dir": (list_dir_tool, LIST_DIR_DEF),
            "apply_patch": (apply_patch, APPLY_PATCH_DEF),
            "spawn_agent": (spawn_agent, SPAWN_AGENT_DEF),
            "send_input": (send_input, SEND_INPUT_DEF),
            "wait": (wait_agent, WAIT_DEF),
            "close_agent": (close_agent, CLOSE_AGENT_DEF),
        }

        for name in tool_names:
            pair = _tool_map.get(name)
            if pair:
                registry.register(name, pair[0], pair[1])

        return registry

    async def submit(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Submit user input and yield events as the agent works.

        This is the main entry point. The session processes the input
        through the agentic loop, emitting events for every step.

        Accepts submissions from IDLE or AWAITING_INPUT states.
        AWAITING_INPUT → PROCESSING is the normal transition when the
        user responds to a model question.
        """
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")
        if self._state == SessionState.PROCESSING:
            raise RuntimeError("Session is already processing")

        self._state = SessionState.PROCESSING
        emitter = EventEmitter()

        emitter.emit(
            self._event(AgentEventType.SESSION_START, {"state": self._state.value})
        )

        # Use profile timeout unless the session config overrides
        effective_timeout = (
            self._config.default_command_timeout_ms
            if self._config.default_command_timeout_ms is not None
            else self._profile.default_timeout_ms
        )

        loop_config = LoopConfig(
            max_turns=self._config.max_turns or None,
            max_tool_rounds=self._config.max_tool_rounds_per_input or None,
            enable_loop_detection=self._config.enable_loop_detection,
            loop_detection_window=self._config.loop_detection_window,
            reasoning_effort=self._reasoning_effort,
            default_command_timeout_ms=effective_timeout,
            max_command_timeout_ms=self._config.max_command_timeout_ms,
            truncation_config=self._config.truncation_config,
            model_id=self._config.model_id,
            user_instructions=self._config.user_instructions,
            context_window_size=self._profile.context_window_size,
            context_window_warning_threshold=self._config.context_window_warning_threshold,
        )

        loop = AgentLoop(
            profile=self._profile,
            environment=self._env,
            registry=self._registry,
            llm_client=self._llm,
            emitter=emitter,
            config=loop_config,
            loop_detector=self._loop_detector,
            abort_event=self._abort_event,
        )
        self._current_loop = loop

        # Run the loop in a background task so we can yield events as
        # they arrive.
        async def _run_loop() -> None:
            try:
                await loop.run(user_input, self._history)

                # Process follow-up queue
                while self._follow_up_queue and not self._abort_event.is_set():
                    follow_up = self._follow_up_queue.pop(0)
                    await loop.run(follow_up, self._history)

            except Exception as exc:
                emitter.emit(
                    self._event(
                        AgentEventType.ERROR,
                        {"error": str(exc), "phase": "session"},
                    )
                )
            finally:
                self._current_loop = None

                if self._abort_event.is_set():
                    self._state = SessionState.CLOSED
                elif self._follow_up_queue:
                    # Follow-ups remain — stay IDLE for next submit
                    self._state = SessionState.IDLE
                else:
                    # Natural completion with no follow-ups queued:
                    # return to IDLE per spec (PROCESSING -> IDLE).
                    self._state = SessionState.IDLE

                emitter.emit(
                    self._event(
                        AgentEventType.SESSION_END,
                        {"state": self._state.value},
                    )
                )
                emitter.close()

        task = asyncio.create_task(_run_loop())

        pending_tool_results: list[ToolResult] = []

        async for event in emitter:
            # Track typed turns from events
            self._record_turn_from_event(event, pending_tool_results)
            yield event

        # Flush any remaining tool results
        if pending_tool_results:
            self._turns.append(ToolResultsTurn(results=list(pending_tool_results)))
            pending_tool_results.clear()

        # Ensure the task is fully done
        await task

    def _record_turn_from_event(
        self,
        event: AgentEvent,
        pending_tool_results: list[ToolResult],
    ) -> None:
        """Build typed Turn records from agent events.

        Tool results are batched: TOOL_CALL_END events accumulate in
        *pending_tool_results* until a non-tool event flushes them into
        a single ToolResultsTurn.
        """
        # Tool-related events: accumulate results, skip non-terminal events
        _TOOL_EVENTS = {
            AgentEventType.TOOL_CALL_START,
            AgentEventType.TOOL_CALL_OUTPUT_DELTA,
            AgentEventType.TOOL_CALL_END,
        }

        if event.type == AgentEventType.TOOL_CALL_END:
            pending_tool_results.append(
                ToolResult(
                    output=event.data.get("truncated_output", ""),
                    is_error=bool(event.data.get("is_error", False)),
                    full_output=event.data.get("output", ""),
                )
            )
            return

        if event.type in _TOOL_EVENTS:
            return

        # Flush pending tool results on any non-tool event
        if pending_tool_results:
            self._turns.append(
                ToolResultsTurn(results=list(pending_tool_results))
            )
            pending_tool_results.clear()

        if event.type == AgentEventType.USER_INPUT:
            self._turns.append(
                UserTurn(content=event.data.get("text", ""))
            )
        elif event.type == AgentEventType.ASSISTANT_TEXT_END:
            self._turns.append(
                AssistantTurn(content=event.data.get("text", ""))
            )
        elif event.type == AgentEventType.STEERING_INJECTED:
            self._turns.append(
                SteeringTurn(content=event.data.get("text", ""))
            )

    def steer(self, message: str) -> None:
        """Queue a steering message to inject after the current tool round.

        Steering messages are injected as user messages between tool rounds,
        allowing the caller to redirect the agent mid-execution.

        Args:
            message: The steering text to inject into the conversation.
        """
        if self._current_loop is not None:
            self._current_loop.queue_steering(message)

    def follow_up(self, message: str) -> None:
        """Queue a message for processing after the current input completes."""
        self._follow_up_queue.append(message)

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Change reasoning effort for the next LLM call."""
        self._reasoning_effort = effort

    def abort(self) -> None:
        """Abort the session: cancel the running loop and transition to CLOSED.

        Sets the abort event so the loop exits at the next check point.
        If the session is not currently processing, the state transitions
        to CLOSED immediately.
        """
        self._abort_event.set()
        if self._state != SessionState.PROCESSING:
            self._state = SessionState.CLOSED

    async def shutdown(self) -> None:
        """Graceful shutdown — close environment and mark session closed."""
        self._state = SessionState.CLOSED
        await self._env.cleanup()
