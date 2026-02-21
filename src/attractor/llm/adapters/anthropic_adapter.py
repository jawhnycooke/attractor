"""Anthropic provider adapter."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from attractor.llm.models import (
    ContentPart,
    FinishReason,
    ImageContent,
    Message,
    ReasoningEffort,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    RedactedThinkingContent,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_FINISH_MAP: dict[str, FinishReason] = {
    "end_turn": FinishReason("stop", raw="end_turn"),
    "tool_use": FinishReason("tool_calls", raw="tool_use"),
    "max_tokens": FinishReason("length", raw="max_tokens"),
    "stop_sequence": FinishReason("stop", raw="stop_sequence"),
}

_THINKING_BUDGET = {
    ReasoningEffort.LOW: 2048,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 32768,
}


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    def provider_name(self) -> str:
        """Return the canonical provider name.

        Returns:
            The string ``"anthropic"``.
        """
        return "anthropic"

    def detect_model(self, model: str) -> bool:
        """Check whether *model* belongs to this provider.

        Args:
            model: Model identifier string (e.g. ``"claude-sonnet-4-20250514"``).

        Returns:
            True if the model string starts with ``"claude-"``.
        """
        return model.startswith("claude-")

    # -----------------------------------------------------------------
    # Request mapping
    # -----------------------------------------------------------------

    def _ensure_alternation(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Anthropic requires strict user/assistant alternation.

        Insert synthetic placeholder messages where consecutive same-role
        messages would otherwise violate the constraint.
        """
        if not messages:
            return messages

        result: list[dict[str, Any]] = [messages[0]]
        for msg in messages[1:]:
            prev_role = result[-1]["role"]
            curr_role = msg["role"]
            if curr_role == prev_role:
                filler_role = "user" if curr_role == "assistant" else "assistant"
                result.append({"role": filler_role, "content": "..."})
            result.append(msg)

        if result and result[0]["role"] != "user":
            result.insert(0, {"role": "user", "content": "..."})

        return result

    def _map_messages(self, request: Request) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for msg in request.messages:
            mapped = self._map_message(msg)
            if mapped is not None:
                msgs.append(mapped)
        return self._ensure_alternation(msgs)

    def _map_message(self, msg: Message) -> dict[str, Any] | None:
        if msg.role == Role.SYSTEM:
            return None  # system handled separately

        if msg.role == Role.TOOL:
            results = []
            for part in msg.content:
                if isinstance(part, ToolResultContent):
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": part.content,
                            **({"is_error": True} if part.is_error else {}),
                        }
                    )
            return {"role": "user", "content": results}

        content = self._map_content_parts(msg)
        role = "assistant" if msg.role == Role.ASSISTANT else "user"
        return {"role": role, "content": content}

    def _map_content_parts(self, msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, TextContent):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if part.base64_data:
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type,
                                "data": part.base64_data,
                            },
                        }
                    )
                elif part.url:
                    parts.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.url},
                        }
                    )
            elif isinstance(part, ToolCallContent):
                args_str = part.arguments_json or json.dumps(part.arguments)
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                except json.JSONDecodeError:
                    args = {}
                parts.append(
                    {
                        "type": "tool_use",
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "input": args,
                    }
                )
            elif isinstance(part, ThinkingContent):
                block: dict[str, Any] = {"type": "thinking", "thinking": part.text}
                if part.signature:
                    block["signature"] = part.signature
                parts.append(block)
            elif isinstance(part, RedactedThinkingContent):
                parts.append({"type": "redacted_thinking", "data": part.data})
        return parts or [{"type": "text", "text": "(empty)"}]

    def _map_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.to_json_schema(),
            }
            for t in tools
        ]

    def _inject_cache_control(
        self, kwargs: dict[str, Any], request: Request
    ) -> None:
        """Auto-inject cache_control on system prompt and long tool definitions.

        Anthropic requires explicit cache_control breakpoints for prompt
        caching. Without them, every turn re-processes the full system prompt
        and conversation history at full price. This method automatically
        adds cache_control to the system prompt and to tool definitions that
        exceed a length threshold (4096 chars in their JSON representation).
        """
        # Check if auto-caching is disabled via provider_options
        if request.provider_options:
            anthropic_opts = request.provider_options.get("anthropic", {})
            if isinstance(anthropic_opts, dict) and not anthropic_opts.get(
                "auto_cache", True
            ):
                return

        # Inject cache_control on system prompt
        if "system" in kwargs and isinstance(kwargs["system"], str):
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": kwargs["system"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Inject cache_control on long tool definitions
        tools = kwargs.get("tools")
        if tools and isinstance(tools, list):
            for tool in tools:
                desc = tool.get("description", "")
                schema = tool.get("input_schema", {})
                # Mark tools with long descriptions or complex schemas
                total_len = len(desc) + len(json.dumps(schema))
                if total_len > 4096:
                    tool["cache_control"] = {"type": "ephemeral"}

    def _build_kwargs(self, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._map_messages(request),
            "max_tokens": request.max_tokens or 4096,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        tools = self._map_tools(request.tools)
        if tools:
            kwargs["tools"] = tools
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        if request.reasoning_effort:
            budget = _THINKING_BUDGET.get(request.reasoning_effort, 8192)
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }

        # Auto-inject cache_control for prompt caching
        self._inject_cache_control(kwargs, request)

        return kwargs

    # -----------------------------------------------------------------
    # Response mapping
    # -----------------------------------------------------------------

    def _map_response(self, raw: Any, latency_ms: float) -> Response:
        content_parts: list[ContentPart] = []

        for block in raw.content:
            if block.type == "text":
                if block.text:
                    content_parts.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                content_parts.append(
                    ToolCallContent(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        arguments=args,
                        arguments_json=json.dumps(args),
                    )
                )
            elif block.type == "thinking":
                content_parts.append(ThinkingContent(
                    text=block.thinking,
                    signature=getattr(block, "signature", None),
                ))
            elif block.type == "redacted_thinking":
                content_parts.append(
                    RedactedThinkingContent(
                        data=getattr(block, "data", ""),
                    )
                )

        usage = TokenUsage(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            cache_read_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0)
            or 0,
        )

        raw_reason = raw.stop_reason or "end_turn"
        finish = _FINISH_MAP.get(raw_reason, FinishReason("stop", raw=raw_reason))

        return Response(
            message=Message(role=Role.ASSISTANT, content=content_parts),
            model=raw.model,
            provider="anthropic",
            finish_reason=finish,
            usage=usage,
            provider_response_id=raw.id or "",
            latency_ms=latency_ms,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        """Send a non-streaming completion request to the Anthropic Messages API.

        Args:
            request: Provider-agnostic request to send.

        Returns:
            Mapped Response with content, usage, and finish reason.
        """
        kwargs = self._build_kwargs(request)
        t0 = time.monotonic()
        raw = await self._client.messages.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000
        return self._map_response(raw, latency)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request to the Anthropic Messages API.

        Args:
            request: Provider-agnostic request to send.

        Yields:
            StreamEvent instances as they arrive from the API.
        """
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        t0 = time.monotonic()
        raw_stream = await self._client.messages.create(**kwargs)

        yield StreamEvent(type=StreamEventType.STREAM_START)

        current_tool: ToolCallContent | None = None

        async for event in raw_stream:
            event_type = event.type

            if event_type == "content_block_start":
                block = event.content_block
                if block.type == "text":
                    pass  # text comes via deltas
                elif block.type == "tool_use":
                    current_tool = ToolCallContent(
                        tool_call_id=block.id,
                        tool_name=block.name,
                    )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call=current_tool,
                    )
                elif block.type == "thinking":
                    pass  # thinking comes via deltas

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text=delta.text,
                    )
                elif delta.type == "input_json_delta":
                    if current_tool:
                        current_tool.arguments_json += delta.partial_json
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            text=delta.partial_json,
                            tool_call=current_tool,
                        )
                elif delta.type == "thinking_delta":
                    yield StreamEvent(
                        type=StreamEventType.REASONING_DELTA,
                        text=delta.thinking,
                    )

            elif event_type == "content_block_stop":
                if current_tool:
                    if current_tool.arguments_json:
                        try:
                            current_tool.arguments = json.loads(
                                current_tool.arguments_json
                            )
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "Failed to parse tool call arguments: %s",
                                exc,
                            )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        tool_call=current_tool,
                    )
                    current_tool = None

            elif event_type == "message_delta":
                raw_sr = getattr(event.delta, "stop_reason", None) or "end_turn"
                finish = _FINISH_MAP.get(
                    raw_sr,
                    FinishReason("stop", raw=raw_sr),
                )
                usage_data = getattr(event, "usage", None)
                usage = None
                if usage_data:
                    usage = TokenUsage(
                        output_tokens=getattr(usage_data, "output_tokens", 0),
                    )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=finish,
                    usage=usage,
                    metadata={"latency_ms": (time.monotonic() - t0) * 1000},
                )

            elif event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg and hasattr(msg, "usage"):
                    yield StreamEvent(
                        type=StreamEventType.STREAM_START,
                        usage=TokenUsage(
                            input_tokens=msg.usage.input_tokens,
                        ),
                    )
