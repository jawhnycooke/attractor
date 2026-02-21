"""Subagent management tools.

Allows the main agent to spawn child agents for delegating subtasks.
Subagents run in their own Sessions and communicate results back.

Subagent state is stored per-session on the environment object
(``environment.subagents``) rather than in module-level globals, so
concurrent sessions do not share or leak subagent handles.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class SubagentHandle:
    """Tracks a running subagent.

    Attributes:
        agent_id: Unique identifier for the subagent.
        task: The asyncio task executing the subagent, if any.
        session: The child Session instance, if created.
        result: Completed result text, or None if still running.
        closed: Whether the subagent has been closed.
        events: Collected events from the subagent run.
    """

    agent_id: str
    task: asyncio.Task[str] | None = None
    session: Any = None
    result: str | None = None
    closed: bool = False
    events: list[Any] = field(default_factory=list)


def _get_subagents(environment: Any) -> dict[str, SubagentHandle]:
    """Return the session-scoped subagent registry from *environment*.

    Falls back to creating the attribute if not present, for
    environments that don't pre-initialise the dict.
    """
    if not hasattr(environment, "subagents"):
        environment.subagents = {}
    return environment.subagents


def _get_session_factory(environment: Any) -> Any | None:
    """Return the session factory from the environment, if available.

    The parent Session sets ``environment._session_factory`` to a callable
    that creates child Sessions with inherited configuration.
    """
    return getattr(environment, "_session_factory", None)


async def _run_subagent(handle: SubagentHandle, task_desc: str) -> str:
    """Run a subagent session and collect the final text output."""
    session = handle.session
    if session is None:
        return f"Subagent '{handle.agent_id}' has no session"

    collected_text: list[str] = []
    try:
        async for event in session.submit(task_desc):
            handle.events.append(event)
            if hasattr(event, "type"):
                type_val = event.type.value if hasattr(event.type, "value") else str(event.type)
                if type_val == "assistant_text_end":
                    text = event.data.get("text", "")
                    if text:
                        collected_text.append(text)
    except Exception as exc:
        return f"Subagent '{handle.agent_id}' failed: {exc}"

    result = "\n".join(collected_text) if collected_text else "Subagent completed with no text output"
    handle.result = result
    return result


async def spawn_agent(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Spawn a new subagent to work on a task.

    Creates a child Session that shares the parent's execution environment
    and runs the task asynchronously. The subagent maintains its own
    conversation history.

    Args:
        arguments: Must contain a ``task`` key describing the work.
        environment: Execution environment (carries session-scoped state).

    Returns:
        ToolResult with the new agent ID.
    """
    task_desc = arguments.get("task", "")
    if not task_desc:
        return ToolResult(
            output="Error: 'task' is required",
            is_error=True,
            full_output="Error: 'task' is required",
        )

    subagents = _get_subagents(environment)
    agent_id = f"subagent_{uuid.uuid4().hex[:8]}"
    handle = SubagentHandle(agent_id=agent_id)

    # Try to create a real child Session via the session factory
    factory = _get_session_factory(environment)
    if factory is not None:
        try:
            model = arguments.get("model")
            max_turns = arguments.get("max_turns", 0)
            child_session = factory(
                model_override=model,
                max_turns_override=max_turns,
            )
            handle.session = child_session

            # Run the subagent in a background task
            handle.task = asyncio.create_task(
                _run_subagent(handle, task_desc)
            )

            subagents[agent_id] = handle
            msg = (
                f"Subagent '{agent_id}' spawned for task: {task_desc}\n"
                f"Use wait(agent_id='{agent_id}') to get results."
            )
            return ToolResult(output=msg, full_output=msg)
        except Exception as exc:
            logger.warning(
                "Failed to create child session for subagent: %s", exc
            )

    # Fallback: create a handle without a real session
    subagents[agent_id] = handle
    msg = (
        f"Subagent '{agent_id}' created for task: {task_desc}\n"
        f"Use wait(agent_id='{agent_id}') to get results."
    )
    return ToolResult(output=msg, full_output=msg)


async def send_input(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Send a follow-up message to a running subagent.

    If the subagent has a real Session, the message is queued via
    ``session.follow_up()``.

    Args:
        arguments: Must contain ``agent_id`` and ``message`` keys.
        environment: Execution environment.

    Returns:
        ToolResult confirming delivery or describing the error.
    """
    agent_id: str = arguments.get("agent_id", "")
    message: str = arguments.get("message", "")

    subagents = _get_subagents(environment)
    handle = subagents.get(agent_id)
    if handle is None:
        return ToolResult(
            output=f"Error: no subagent with id '{agent_id}'",
            is_error=True,
            full_output=f"Error: no subagent with id '{agent_id}'",
        )
    if handle.closed:
        return ToolResult(
            output=f"Error: subagent '{agent_id}' is closed",
            is_error=True,
            full_output=f"Error: subagent '{agent_id}' is closed",
        )

    # Deliver to real session if available
    if handle.session is not None and hasattr(handle.session, "follow_up"):
        handle.session.follow_up(message)
        msg = f"Message delivered to subagent '{agent_id}'"
    else:
        msg = f"Message queued for subagent '{agent_id}'"

    return ToolResult(output=msg, full_output=msg)


async def wait_agent(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Wait for a subagent to complete and return its result.

    Args:
        arguments: Must contain an ``agent_id`` key.
        environment: Execution environment.

    Returns:
        ToolResult with the subagent output or an error.
    """
    agent_id: str = arguments.get("agent_id", "")

    subagents = _get_subagents(environment)
    handle = subagents.get(agent_id)
    if handle is None:
        return ToolResult(
            output=f"Error: no subagent with id '{agent_id}'",
            is_error=True,
            full_output=f"Error: no subagent with id '{agent_id}'",
        )

    if handle.result is not None:
        return ToolResult(output=handle.result, full_output=handle.result)

    if handle.task is not None:
        try:
            result = await handle.task
            handle.result = result
            return ToolResult(output=result, full_output=result)
        except asyncio.CancelledError:
            msg = f"Subagent '{agent_id}' was cancelled"
            return ToolResult(output=msg, is_error=True, full_output=msg)
        except Exception as exc:
            msg = f"Subagent '{agent_id}' failed: {exc}"
            return ToolResult(output=msg, is_error=True, full_output=msg)

    msg = f"Subagent '{agent_id}' has no result yet"
    return ToolResult(output=msg, full_output=msg)


async def close_agent(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Close a subagent and free its resources.

    If the subagent has a real Session, it is shut down gracefully.

    Args:
        arguments: Must contain an ``agent_id`` key.
        environment: Execution environment.

    Returns:
        ToolResult confirming closure or describing the error.
    """
    agent_id: str = arguments.get("agent_id", "")

    subagents = _get_subagents(environment)
    handle = subagents.get(agent_id)
    if handle is None:
        return ToolResult(
            output=f"Error: no subagent with id '{agent_id}'",
            is_error=True,
            full_output=f"Error: no subagent with id '{agent_id}'",
        )

    handle.closed = True

    # Shut down child session if present
    if handle.session is not None and hasattr(handle.session, "shutdown"):
        try:
            await handle.session.shutdown()
        except Exception as exc:
            logger.warning("Error shutting down subagent session: %s", exc)

    if handle.task and not handle.task.done():
        handle.task.cancel()
        try:
            await handle.task
        except (asyncio.CancelledError, Exception):
            pass

    del subagents[agent_id]
    msg = f"Subagent '{agent_id}' closed"
    return ToolResult(output=msg, full_output=msg)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

SPAWN_AGENT_DEF = ToolDefinition(
    name="spawn_agent",
    description="Spawn a subagent to work on a delegated task.",
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Description of the task for the subagent.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the subagent.",
            },
            "model": {
                "type": "string",
                "description": "Model to use for the subagent.",
            },
            "max_turns": {
                "type": "integer",
                "description": "Maximum turns for the subagent.",
            },
        },
        "required": ["task"],
    },
)

SEND_INPUT_DEF = ToolDefinition(
    name="send_input",
    description="Send a follow-up message to a running subagent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The subagent ID.",
            },
            "message": {
                "type": "string",
                "description": "The message to send.",
            },
        },
        "required": ["agent_id", "message"],
    },
)

WAIT_DEF = ToolDefinition(
    name="wait",
    description="Wait for a subagent to complete and get its result.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The subagent ID to wait on.",
            },
        },
        "required": ["agent_id"],
    },
)

CLOSE_AGENT_DEF = ToolDefinition(
    name="close_agent",
    description="Close a subagent and free its resources.",
    parameters={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The subagent ID to close.",
            },
        },
        "required": ["agent_id"],
    },
)
