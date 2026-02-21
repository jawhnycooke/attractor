"""Streaming utilities for the Unified LLM Client.

Provides StreamCollector to accumulate streaming events into a complete
Response, handling partial tool call assembly and token usage tracking.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from attractor.llm.models import (
    ContentPart,
    FinishReason,
    Message,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCallContent,
    TokenUsage,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamCollector:
    """Accumulates StreamEvents into a complete Response.

    Feed events via `process_event()` or collect an entire async stream
    with `collect()`. After all events have been processed, call
    `to_response()` to get the assembled Response.
    """

    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallContent] = field(default_factory=list)
    _pending_tool: ToolCallContent | None = field(default=None, repr=False)
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def process_event(self, event: StreamEvent) -> None:
        """Process a single stream event."""
        if event.type == StreamEventType.TEXT_DELTA:
            self.text_parts.append(event.text)

        elif event.type == StreamEventType.TOOL_CALL_START:
            if event.tool_call:
                self._pending_tool = ToolCallContent(
                    tool_call_id=event.tool_call.tool_call_id,
                    tool_name=event.tool_call.tool_name,
                )

        elif event.type == StreamEventType.TOOL_CALL_DELTA:
            if self._pending_tool and event.text:
                self._pending_tool.arguments_json += event.text

        elif event.type == StreamEventType.TOOL_CALL_END:
            if event.tool_call:
                tc = event.tool_call
                if tc.arguments_json and not tc.arguments:
                    try:
                        tc.arguments = json.loads(tc.arguments_json)
                    except json.JSONDecodeError as exc:
                        logger.warning("Failed to parse tool call arguments: %s", exc)
                self.tool_calls.append(tc)
                self._pending_tool = None
            elif self._pending_tool:
                if self._pending_tool.arguments_json:
                    try:
                        self._pending_tool.arguments = json.loads(
                            self._pending_tool.arguments_json
                        )
                    except json.JSONDecodeError as exc:
                        logger.warning("Failed to parse tool call arguments: %s", exc)
                self.tool_calls.append(self._pending_tool)
                self._pending_tool = None

        elif event.type == StreamEventType.FINISH:
            if event.finish_reason:
                self.finish_reason = event.finish_reason
            if event.usage:
                self._merge_usage(event.usage)
            if event.metadata:
                self.metadata.update(event.metadata)

        elif event.type == StreamEventType.STREAM_START:
            if event.usage:
                self._merge_usage(event.usage)

    def _merge_usage(self, usage: TokenUsage) -> None:
        if usage.input_tokens:
            self.usage.input_tokens = usage.input_tokens
        if usage.output_tokens:
            self.usage.output_tokens = usage.output_tokens
        if usage.reasoning_tokens:
            self.usage.reasoning_tokens = usage.reasoning_tokens
        if usage.cache_read_tokens:
            self.usage.cache_read_tokens = usage.cache_read_tokens
        if usage.cache_write_tokens:
            self.usage.cache_write_tokens = usage.cache_write_tokens

    def to_response(self) -> Response:
        """Assemble accumulated events into a complete Response."""
        content_parts: list[ContentPart] = []

        full_text = "".join(self.text_parts)
        if full_text:
            content_parts.append(TextContent(text=full_text))

        content_parts.extend(self.tool_calls)

        return Response(
            message=Message(role=Role.ASSISTANT, content=content_parts),
            model=self.model,
            finish_reason=self.finish_reason,
            usage=self.usage,
            latency_ms=self.metadata.get("latency_ms", 0.0),
            metadata=self.metadata,
        )

    async def collect(self, stream: AsyncIterator[StreamEvent]) -> Response:
        """Consume an entire async stream and return the assembled Response."""
        async for event in stream:
            self.process_event(event)
        return self.to_response()
