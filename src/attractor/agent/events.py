"""Agent event system for streaming execution progress.

Defines event types and an async EventEmitter that delivers events
to consumers via an async iterator interface.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any


class AgentEventType(str, enum.Enum):
    """Types of events emitted during agent execution."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single event from the agent execution pipeline.

    Args:
        type: The kind of event.
        data: Event payload — contents vary by event type.
        timestamp: Unix timestamp when the event was created.
    """

    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


_SENTINEL = object()


class EventEmitter:
    """Async event emitter with iterator-based delivery.

    Emitted events are placed into an internal queue and consumed
    by async-iterating over the emitter instance.

    Example::

        emitter = EventEmitter()
        emitter.emit(AgentEvent(type=AgentEventType.SESSION_START))

        async for event in emitter:
            print(event.type)
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | object] = asyncio.Queue()
        self._closed = False

    def emit(self, event: AgentEvent) -> None:
        """Enqueue an event for delivery."""
        if not self._closed:
            self._queue.put_nowait(event)

    def close(self) -> None:
        """Signal that no more events will be emitted."""
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_SENTINEL)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self

    async def __anext__(self) -> AgentEvent:
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]
