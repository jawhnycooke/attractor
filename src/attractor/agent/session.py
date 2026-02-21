"""Agent session — the top-level API for driving the coding agent.

A Session wraps an AgentLoop, manages conversation history, and exposes
async-iterator-based event delivery.
"""

from __future__ import annotations

import asyncio
import enum
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
    READ_FILE_DEF,
    SHELL_DEF,
    WRITE_FILE_DEF,
    edit_file,
    glob_tool,
    grep_tool,
    read_file,
    shell,
    write_file,
)
from attractor.agent.tools.apply_patch import APPLY_PATCH_DEF, apply_patch
from attractor.agent.tools.registry import ToolRegistry
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
from attractor.llm.models import Message, ReasoningEffort

# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration for an agent session."""

    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 0
    default_command_timeout_ms: int = 10_000
    reasoning_effort: ReasoningEffort | None = None
    enable_loop_detection: bool = True
    max_subagent_depth: int = 1
    model_id: str = ""
    user_instructions: str = ""
    truncation_config: TruncationConfig | None = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
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
    ) -> None:
        self._profile = profile
        self._env = environment
        self._config = config
        self._llm = llm_client
        self._state = SessionState.IDLE
        self._history: list[Message] = []
        self._follow_up_queue: list[str] = []
        self._reasoning_effort = config.reasoning_effort
        self._registry = self._build_registry()
        self._loop_detector = LoopDetector()
        self._current_loop: AgentLoop | None = None

    @property
    def state(self) -> SessionState:
        """Current session lifecycle state."""
        return self._state

    @property
    def conversation_history(self) -> list[Message]:
        """Snapshot of the conversation history (defensive copy)."""
        return list(self._history)

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
        """
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")

        self._state = SessionState.RUNNING
        emitter = EventEmitter()

        emitter.emit(
            AgentEvent(
                type=AgentEventType.SESSION_START,
                data={"state": self._state.value},
            )
        )

        loop_config = LoopConfig(
            max_turns=self._config.max_turns or None,
            max_tool_rounds=self._config.max_tool_rounds_per_input or None,
            enable_loop_detection=self._config.enable_loop_detection,
            reasoning_effort=self._reasoning_effort,
            default_command_timeout_ms=self._config.default_command_timeout_ms,
            truncation_config=self._config.truncation_config,
            model_id=self._config.model_id,
            user_instructions=self._config.user_instructions,
        )

        loop = AgentLoop(
            profile=self._profile,
            environment=self._env,
            registry=self._registry,
            llm_client=self._llm,
            emitter=emitter,
            config=loop_config,
            loop_detector=self._loop_detector,
        )
        self._current_loop = loop

        # Run the loop in a background task so we can yield events as
        # they arrive.
        async def _run_loop() -> None:
            try:
                await loop.run(user_input, self._history)

                # Process follow-up queue
                while self._follow_up_queue:
                    follow_up = self._follow_up_queue.pop(0)
                    await loop.run(follow_up, self._history)

            except Exception as exc:
                emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        data={"error": str(exc), "phase": "session"},
                    )
                )
            finally:
                self._state = SessionState.IDLE
                self._current_loop = None
                emitter.emit(
                    AgentEvent(
                        type=AgentEventType.SESSION_END,
                        data={"state": self._state.value},
                    )
                )
                emitter.close()

        task = asyncio.create_task(_run_loop())

        async for event in emitter:
            yield event

        # Ensure the task is fully done
        await task

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

    async def shutdown(self) -> None:
        """Graceful shutdown — close environment and mark session closed."""
        self._state = SessionState.CLOSED
        await self._env.cleanup()
