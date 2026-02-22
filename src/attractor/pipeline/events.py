"""Pipeline event system for observability.

Provides typed events emitted during pipeline execution for UI, logging,
and metrics integration per spec Section 9.6.
"""

from __future__ import annotations

import enum
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class PipelineEventType(str, enum.Enum):
    """Typed event categories emitted during pipeline execution."""

    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_RETRY = "node_retry"
    NODE_FAIL = "node_fail"
    CHECKPOINT_SAVED = "checkpoint_saved"
    PARALLEL_STARTED = "parallel_started"
    PARALLEL_BRANCH_STARTED = "parallel_branch_started"
    PARALLEL_BRANCH_COMPLETED = "parallel_branch_completed"
    PARALLEL_COMPLETED = "parallel_completed"
    INTERVIEW_STARTED = "interview_started"
    INTERVIEW_COMPLETED = "interview_completed"
    INTERVIEW_TIMEOUT = "interview_timeout"


@dataclass
class PipelineEvent:
    """A single pipeline lifecycle event.

    Attributes:
        type: The event category.
        node_name: Name of the relevant node (empty for pipeline-level events).
        pipeline_name: Name of the pipeline emitting this event.
        timestamp: UNIX epoch when the event occurred.
        data: Arbitrary event-specific payload.
    """

    type: PipelineEventType
    node_name: str = ""
    pipeline_name: str = ""
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


# Callback type: async function that receives a PipelineEvent
EventCallback = Callable[[PipelineEvent], Coroutine[Any, Any, None]]


class PipelineEventEmitter:
    """Observer-pattern event emitter for pipeline lifecycle events.

    Register callbacks with :meth:`on` and fire events with :meth:`emit`.
    """

    def __init__(self) -> None:
        self._listeners: dict[PipelineEventType, list[EventCallback]] = defaultdict(
            list
        )

    @property
    def listeners(self) -> dict[PipelineEventType, list[EventCallback]]:
        """Return the mapping of event types to registered callbacks."""
        return dict(self._listeners)

    def on(self, event_type: PipelineEventType, callback: EventCallback) -> None:
        """Register a callback for a specific event type.

        Args:
            event_type: The event category to listen for.
            callback: Async callable invoked when the event fires.
        """
        self._listeners[event_type].append(callback)

    async def emit(self, event: PipelineEvent) -> None:
        """Fire an event, invoking all registered callbacks.

        Exceptions in callbacks are logged but do not prevent
        other callbacks from running.

        Args:
            event: The event to emit.
        """
        for callback in self._listeners.get(event.type, []):
            try:
                await callback(event)
            except Exception as exc:
                logger.error(
                    "Event callback error for %s: %s",
                    event.type.value,
                    exc,
                )
