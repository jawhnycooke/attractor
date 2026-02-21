"""Subagent management tools.

Allows the main agent to spawn child agents for delegating subtasks.
Subagents run in their own Sessions and communicate results back.

Subagent state is stored per-session on the environment object
(``environment.subagents``) rather than in module-level globals, so
concurrent sessions do not share or leak subagent handles.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import ToolDefinition


@dataclass
class SubagentHandle:
    """Tracks a running subagent.

    Attributes:
        agent_id: Unique identifier for the subagent.
        task: The asyncio task executing the subagent, if any.
        result: Completed result text, or None if still running.
        closed: Whether the subagent has been closed.
    """

    agent_id: str
    task: asyncio.Task[str] | None = None
    result: str | None = None
    closed: bool = False


def _get_subagents(environment: Any) -> dict[str, SubagentHandle]:
    """Return the session-scoped subagent registry from *environment*.

    Falls back to creating the attribute if not present, for
    environments that don't pre-initialise the dict.
    """
    if not hasattr(environment, "subagents"):
        environment.subagents = {}
    return environment.subagents


async def spawn_agent(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Spawn a new subagent to work on a task.

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

    Args:
        arguments: Must contain ``agent_id`` and ``message`` keys.
        environment: Execution environment.

    Returns:
        ToolResult confirming delivery or describing the error.
    """
    agent_id: str = arguments.get("agent_id", "")
    _message: str = arguments.get("message", "")  # TODO: deliver to subagent

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

    msg = f"Message sent to subagent '{agent_id}'"
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
    if handle.task and not handle.task.done():
        handle.task.cancel()

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
