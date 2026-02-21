"""Core agentic loop — the engine that drives tool-call cycles.

The AgentLoop executes the standard LLM → tool → LLM loop:
1. Build a Request from history, system prompt, and tools
2. Call the LLM
3. If the response contains tool calls, execute them, append results, loop
4. If text-only, return (natural completion)
5. If turn limit hit, emit event and return
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.events import AgentEvent, AgentEventType, EventEmitter
from attractor.agent.loop_detection import LoopDetector
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.prompts import build_system_prompt
from attractor.agent.tools.registry import ToolRegistry
from attractor.agent.truncation import TruncationConfig, truncate_output
from attractor.llm.models import (
    Message,
    ReasoningEffort,
    Request,
    Response,
    Role,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


# ---------------------------------------------------------------------------
# LLM client protocol (decouples from concrete implementation)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal interface the loop needs from an LLM client."""

    async def complete(self, request: Request) -> Response: ...


# ---------------------------------------------------------------------------
# Loop configuration
# ---------------------------------------------------------------------------

class LoopConfig:
    """Configuration passed into the loop from Session."""

    def __init__(
        self,
        max_turns: int = 0,
        max_tool_rounds: int = 0,
        enable_loop_detection: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
        default_command_timeout_ms: int = 10_000,
        truncation_config: TruncationConfig | None = None,
        model_id: str = "",
        user_instructions: str = "",
    ) -> None:
        self.max_turns = max_turns
        self.max_tool_rounds = max_tool_rounds
        self.enable_loop_detection = enable_loop_detection
        self.reasoning_effort = reasoning_effort
        self.default_command_timeout_ms = default_command_timeout_ms
        self.truncation_config = truncation_config
        self.model_id = model_id
        self.user_instructions = user_instructions


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Executes the agentic tool-use loop.

    This class is stateless between runs — all conversation state lives in
    the ``history`` list passed in from Session.
    """

    def __init__(
        self,
        profile: ProviderProfile,
        environment: ExecutionEnvironment,
        registry: ToolRegistry,
        llm_client: LLMClientProtocol,
        emitter: EventEmitter,
        config: LoopConfig,
        loop_detector: LoopDetector,
    ) -> None:
        self._profile = profile
        self._env = environment
        self._registry = registry
        self._llm = llm_client
        self._emitter = emitter
        self._config = config
        self._detector = loop_detector
        self._steering_queue: list[str] = []

    def queue_steering(self, message: str) -> None:
        """Queue a steering message for injection after the current tool round."""
        self._steering_queue.append(message)

    async def run(self, user_input: str, history: list[Message]) -> None:
        """Execute the agentic loop for a single user input.

        Modifies *history* in place by appending assistant and tool messages.
        """
        # Append user message
        history.append(Message.user(user_input))
        self._emitter.emit(AgentEvent(
            type=AgentEventType.USER_INPUT,
            data={"text": user_input},
        ))

        system_prompt = build_system_prompt(
            self._profile,
            self._env,
            model_id=self._config.model_id,
            user_instructions=self._config.user_instructions,
        )
        tools = self._profile.get_tools()

        turn = 0
        while True:
            # Check turn limit
            if self._config.max_turns > 0 and turn >= self._config.max_turns:
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.TURN_LIMIT,
                    data={"turns": turn, "limit": self._config.max_turns},
                ))
                return

            if (
                self._config.max_tool_rounds > 0
                and turn >= self._config.max_tool_rounds
            ):
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.TURN_LIMIT,
                    data={"turns": turn, "limit": self._config.max_tool_rounds},
                ))
                return

            # Build request
            request = Request(
                messages=list(history),
                system_prompt=system_prompt,
                tools=tools,
                reasoning_effort=self._config.reasoning_effort,
                model=self._config.model_id,
            )

            # Call LLM
            try:
                response = await self._llm.complete(request)
            except Exception as exc:
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.ERROR,
                    data={"error": str(exc), "phase": "llm_call"},
                ))
                return

            # Record assistant message
            history.append(response.message)

            # Check for tool calls
            tool_calls = response.message.tool_calls()
            if not tool_calls:
                # Text-only response — natural completion
                text = response.message.text()
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.ASSISTANT_TEXT_START,
                    data={},
                ))
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.ASSISTANT_TEXT_DELTA,
                    data={"text": text},
                ))
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.ASSISTANT_TEXT_END,
                    data={"text": text},
                ))
                return

            # Execute tool calls
            for tc in tool_calls:
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.TOOL_CALL_START,
                    data={
                        "tool_name": tc.tool_name,
                        "tool_call_id": tc.tool_call_id,
                        "arguments": tc.arguments,
                    },
                ))

                result = await self._registry.dispatch(
                    tc.tool_name, tc.arguments, self._env
                )

                # Apply truncation
                truncated, full = truncate_output(
                    tc.tool_name,
                    result.output,
                    config=self._config.truncation_config,
                )
                result.output = truncated
                if not result.full_output:
                    result.full_output = full

                # Emit output delta
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.TOOL_CALL_OUTPUT_DELTA,
                    data={
                        "tool_call_id": tc.tool_call_id,
                        "output": truncated,
                    },
                ))

                # TOOL_CALL_END carries the full untruncated output
                self._emitter.emit(AgentEvent(
                    type=AgentEventType.TOOL_CALL_END,
                    data={
                        "tool_name": tc.tool_name,
                        "tool_call_id": tc.tool_call_id,
                        "output": truncated,
                        "full_output": full,
                        "is_error": result.is_error,
                    },
                ))

                # Append tool result to history
                history.append(Message.tool_result(
                    tool_call_id=tc.tool_call_id,
                    content=truncated,
                    is_error=result.is_error,
                ))

                # Record for loop detection
                if self._config.enable_loop_detection:
                    args_hash = hashlib.sha256(
                        json.dumps(tc.arguments, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    self._detector.record_call(tc.tool_name, args_hash)

            # Inject queued steering messages
            if self._steering_queue:
                for msg in self._steering_queue:
                    history.append(Message.user(msg))
                    self._emitter.emit(AgentEvent(
                        type=AgentEventType.STEERING_INJECTED,
                        data={"text": msg},
                    ))
                self._steering_queue.clear()

            # Check for loops
            if self._config.enable_loop_detection:
                warning = self._detector.check_for_loops()
                if warning:
                    self._emitter.emit(AgentEvent(
                        type=AgentEventType.LOOP_DETECTION,
                        data={"warning": warning},
                    ))
                    # Inject the warning as a user message so the model sees it
                    history.append(Message.user(
                        f"[SYSTEM WARNING] {warning}"
                    ))

            turn += 1
