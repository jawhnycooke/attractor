"""Core agentic loop — the engine that drives tool-call cycles.

The AgentLoop executes the standard LLM → tool → LLM loop:
1. Build a Request from history, system prompt, and tools
2. Call the LLM
3. If the response contains tool calls, execute them, append results, loop
4. If text-only, return (natural completion)
5. If turn limit hit, emit event and return
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
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
)

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class LoopConfig:
    """Immutable configuration passed into the loop from Session.

    Attributes:
        max_turns: Maximum agentic turns (total LLM round-trips), or None
            for unlimited.
        max_tool_rounds: Maximum tool-call rounds per input, or None for
            unlimited.
        enable_loop_detection: Whether to detect repeating tool patterns.
        loop_detection_window: Number of recent tool calls to consider
            when checking for repeating patterns.
        reasoning_effort: Optional reasoning effort level for the LLM.
        default_command_timeout_ms: Default timeout for shell commands.
        max_command_timeout_ms: Upper bound for shell command timeouts.
        truncation_config: Optional truncation settings for tool output.
        model_id: Model identifier string.
        user_instructions: Extra user instructions for the system prompt.
        context_window_size: Context window size in tokens for the model.
        context_window_warning_threshold: Fraction (0.0-1.0) of context
            window usage that triggers a warning event.
    """

    max_turns: int | None = None
    max_tool_rounds: int | None = None
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    reasoning_effort: ReasoningEffort | None = None
    default_command_timeout_ms: int = 10_000
    max_command_timeout_ms: int = 600_000
    truncation_config: TruncationConfig | None = None
    model_id: str = ""
    user_instructions: str = ""
    context_window_size: int = 200_000
    context_window_warning_threshold: float = 0.8


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
        abort_event: asyncio.Event | None = None,
    ) -> None:
        self._profile = profile
        self._env = environment
        self._registry = registry
        self._llm = llm_client
        self._emitter = emitter
        self._config = config
        self._detector = loop_detector
        self._abort_event = abort_event
        self._steering_queue: list[str] = []
        self._total_turns: int = 0

    def queue_steering(self, message: str) -> None:
        """Queue a steering message for injection after the current tool round."""
        self._steering_queue.append(message)

    def _drain_steering(self, history: list[Message]) -> None:
        """Drain all queued steering messages into history."""
        if self._steering_queue:
            for msg in self._steering_queue:
                history.append(Message.user(msg))
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.STEERING_INJECTED,
                        data={"text": msg},
                    )
                )
            self._steering_queue.clear()

    async def _execute_single_tool(
        self,
        tc: Any,
    ) -> tuple[str, str, bool]:
        """Execute a single tool call with truncation.

        Returns:
            Tuple of (truncated_output, full_output, is_error).
        """
        # Validate tool arguments against JSON schema if available
        validation_error = self._validate_tool_arguments(
            tc.tool_name, tc.arguments
        )
        if validation_error:
            return validation_error, validation_error, True

        # Apply default/ceiling timeout for shell tool
        if tc.tool_name == "shell":
            if "timeout_ms" not in tc.arguments:
                tc.arguments["timeout_ms"] = (
                    self._config.default_command_timeout_ms
                )
            else:
                tc.arguments["timeout_ms"] = min(
                    tc.arguments["timeout_ms"],
                    self._config.max_command_timeout_ms,
                )

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

        return truncated, full, result.is_error

    def _validate_tool_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Validate tool arguments against the tool's JSON schema.

        Validates required fields, types, enums, patterns, string length
        constraints, numeric range constraints, array item types, and
        nested object schemas.

        Returns an error message string if validation fails, None if valid.
        """
        definition = self._registry._definitions.get(tool_name)
        if definition is None:
            return None  # Unknown tool — dispatch will handle it

        schema = definition.parameters
        if not schema:
            return None

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required:
            if field_name not in arguments:
                return (
                    f"Missing required argument '{field_name}' "
                    f"for tool '{tool_name}'"
                )

        for field_name, value in arguments.items():
            if field_name in properties:
                prop_schema = properties[field_name]
                error = self._validate_value(
                    value, prop_schema, field_name, tool_name
                )
                if error:
                    return error

        return None

    def _validate_value(
        self,
        value: Any,
        schema: dict[str, Any],
        field_name: str,
        tool_name: str,
    ) -> str | None:
        """Validate a single value against its JSON schema property.

        Checks type, enum, pattern, minLength/maxLength, minimum/maximum,
        array items, and nested object properties.

        Returns an error message string if validation fails, None if valid.
        """
        expected_type = schema.get("type")
        if expected_type and not self._check_json_type(value, expected_type):
            return (
                f"Argument '{field_name}' for tool '{tool_name}' "
                f"expected type '{expected_type}', got "
                f"'{type(value).__name__}'"
            )

        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            return (
                f"Argument '{field_name}' for tool '{tool_name}' "
                f"must be one of {schema['enum']}, got '{value}'"
            )

        # String-specific constraints
        if isinstance(value, str):
            import re

            if "pattern" in schema:
                if not re.search(schema["pattern"], value):
                    return (
                        f"Argument '{field_name}' for tool '{tool_name}' "
                        f"does not match pattern '{schema['pattern']}'"
                    )
            if "minLength" in schema and len(value) < schema["minLength"]:
                return (
                    f"Argument '{field_name}' for tool '{tool_name}' "
                    f"length {len(value)} is below minimum "
                    f"{schema['minLength']}"
                )
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                return (
                    f"Argument '{field_name}' for tool '{tool_name}' "
                    f"length {len(value)} exceeds maximum "
                    f"{schema['maxLength']}"
                )

        # Numeric constraints
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                return (
                    f"Argument '{field_name}' for tool '{tool_name}' "
                    f"value {value} is below minimum {schema['minimum']}"
                )
            if "maximum" in schema and value > schema["maximum"]:
                return (
                    f"Argument '{field_name}' for tool '{tool_name}' "
                    f"value {value} exceeds maximum {schema['maximum']}"
                )

        # Array item type validation
        if isinstance(value, list) and "items" in schema:
            items_schema = schema["items"]
            for i, item in enumerate(value):
                item_error = self._validate_value(
                    item, items_schema, f"{field_name}[{i}]", tool_name
                )
                if item_error:
                    return item_error

        # Nested object validation
        if isinstance(value, dict) and "properties" in schema:
            nested_required = schema.get("required", [])
            nested_props = schema["properties"]
            for req_field in nested_required:
                if req_field not in value:
                    return (
                        f"Missing required field '{req_field}' in "
                        f"'{field_name}' for tool '{tool_name}'"
                    )
            for nested_key, nested_val in value.items():
                if nested_key in nested_props:
                    nested_error = self._validate_value(
                        nested_val,
                        nested_props[nested_key],
                        f"{field_name}.{nested_key}",
                        tool_name,
                    )
                    if nested_error:
                        return nested_error

        return None

    @staticmethod
    def _check_json_type(value: Any, expected: str) -> bool:
        """Check if a value matches a JSON schema type."""
        type_map: dict[str, tuple[type, ...]] = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "object": (dict,),
        }
        allowed = type_map.get(expected)
        if allowed is None:
            return True  # Unknown type — allow
        return isinstance(value, allowed)

    def _check_context_window(self, response: Response) -> None:
        """Emit a warning if context window usage exceeds the threshold."""
        if not response.usage:
            return
        total_used = response.usage.input_tokens + response.usage.output_tokens
        window = self._config.context_window_size
        threshold = self._config.context_window_warning_threshold
        if window > 0 and total_used > window * threshold:
            pct = (total_used / window) * 100
            self._emitter.emit(
                AgentEvent(
                    type=AgentEventType.CONTEXT_WINDOW_WARNING,
                    data={
                        "tokens_used": total_used,
                        "context_window": window,
                        "usage_percent": round(pct, 1),
                    },
                )
            )

    async def run(self, user_input: str, history: list[Message]) -> None:
        """Execute the agentic loop for a single user input.

        Modifies *history* in place by appending assistant and tool messages.
        """
        # Append user message
        history.append(Message.user(user_input))
        self._emitter.emit(
            AgentEvent(
                type=AgentEventType.USER_INPUT,
                data={"text": user_input},
            )
        )

        # Drain any pending steering messages before the first LLM call
        self._drain_steering(history)

        system_prompt = await build_system_prompt(
            self._profile,
            self._env,
            model_id=self._config.model_id,
            user_instructions=self._config.user_instructions,
        )
        tools = self._profile.get_tools()

        tool_round = 0
        while True:
            # Check abort signal
            if self._abort_event and self._abort_event.is_set():
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        data={"error": "Aborted", "phase": "abort"},
                    )
                )
                return

            # Check total turn limit (across all inputs in the session)
            if (
                self._config.max_turns is not None
                and self._total_turns >= self._config.max_turns
            ):
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.TURN_LIMIT,
                        data={
                            "turns": self._total_turns,
                            "limit": self._config.max_turns,
                        },
                    )
                )
                return

            # Check per-input tool round limit
            if (
                self._config.max_tool_rounds is not None
                and tool_round >= self._config.max_tool_rounds
            ):
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.TURN_LIMIT,
                        data={
                            "turns": tool_round,
                            "limit": self._config.max_tool_rounds,
                        },
                    )
                )
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
                # Pop the user message to maintain user/assistant alternation
                # invariant — no assistant response was generated.
                history.pop()
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        data={"error": str(exc), "phase": "llm_call"},
                    )
                )
                return

            self._total_turns += 1

            # Track context window usage
            self._check_context_window(response)

            # Record assistant message
            history.append(response.message)

            # Check for tool calls
            tool_calls = response.message.tool_calls()
            if not tool_calls:
                # Text-only response — natural completion
                text = response.message.text()
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ASSISTANT_TEXT_START,
                        data={},
                    )
                )
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ASSISTANT_TEXT_DELTA,
                        data={"text": text},
                    )
                )
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ASSISTANT_TEXT_END,
                        data={"text": text},
                    )
                )
                return

            # Emit TOOL_CALL_START for all tool calls first
            for tc in tool_calls:
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.TOOL_CALL_START,
                        data={
                            "tool_name": tc.tool_name,
                            "tool_call_id": tc.tool_call_id,
                            "arguments": tc.arguments,
                        },
                    )
                )

            # Execute tool calls concurrently via asyncio.gather
            if len(tool_calls) > 1:
                results = list(
                    await asyncio.gather(
                        *(
                            self._execute_single_tool(tc)
                            for tc in tool_calls
                        )
                    )
                )
            else:
                results = [await self._execute_single_tool(tool_calls[0])]

            # Emit events and append results to history
            for tc, (truncated, full, is_error) in zip(
                tool_calls, results
            ):
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.TOOL_CALL_OUTPUT_DELTA,
                        data={
                            "tool_call_id": tc.tool_call_id,
                            "output": truncated,
                        },
                    )
                )

                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.TOOL_CALL_END,
                        data={
                            "tool_name": tc.tool_name,
                            "tool_call_id": tc.tool_call_id,
                            "output": full,
                            "truncated_output": truncated,
                            "is_error": is_error,
                        },
                    )
                )

                history.append(
                    Message.tool_result(
                        tool_call_id=tc.tool_call_id,
                        content=truncated,
                        is_error=is_error,
                    )
                )

                # Record for loop detection
                if self._config.enable_loop_detection:
                    args_hash = hashlib.sha256(
                        json.dumps(tc.arguments, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    self._detector.record_call(tc.tool_name, args_hash)

            # Inject queued steering messages
            self._drain_steering(history)

            # Check for loops
            if self._config.enable_loop_detection:
                warning = self._detector.check_for_loops(
                    window_size=self._config.loop_detection_window
                )
                if warning:
                    self._emitter.emit(
                        AgentEvent(
                            type=AgentEventType.LOOP_DETECTION,
                            data={"warning": warning},
                        )
                    )
                    # Inject the warning as a user message so the model sees it
                    history.append(Message.user(f"[SYSTEM WARNING] {warning}"))

            # Check abort signal after tool execution
            if self._abort_event and self._abort_event.is_set():
                self._emitter.emit(
                    AgentEvent(
                        type=AgentEventType.ERROR,
                        data={"error": "Aborted", "phase": "abort"},
                    )
                )
                return

            tool_round += 1
