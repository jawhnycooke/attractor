"""Subagent management tools.

Allows the main agent to spawn child agents for delegating subtasks.
Subagents run in their own Sessions and communicate results back.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import ToolDefinition


@dataclass
class SubagentHandle:
    """Tracks a running subagent."""

    agent_id: str
    task: asyncio.Task[str] | None = None
    result: str | None = None
    closed: bool = False


# Module-level registry of active subagents.
_subagents: dict[str, SubagentHandle] = {}


def _register_subagent(handle: SubagentHandle) -> None:
    _subagents[handle.agent_id] = handle


def get_active_subagents() -> dict[str, SubagentHandle]:
    return _subagents


async def spawn_agent(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Spawn a new subagent to work on a task.

    The actual session creation is handled by the parent session
    which injects the spawn logic. This is a placeholder that will be
    wired up by the Session class.
    """
    task_desc = arguments.get("task", "")
    if not task_desc:
        return ToolResult(
            output="Error: 'task' is required",
            is_error=True,
            full_output="Error: 'task' is required",
        )

    agent_id = f"subagent_{uuid.uuid4().hex[:8]}"
    handle = SubagentHandle(agent_id=agent_id)
    _register_subagent(handle)

    msg = (
        f"Subagent '{agent_id}' created for task: {task_desc}\n"
        f"Use wait(agent_id='{agent_id}') to get results."
    )
    return ToolResult(output=msg, full_output=msg)


async def send_input(
    arguments: dict[str, Any],
    environment: Any,
) -> ToolResult:
    """Send a follow-up message to a running subagent."""
    agent_id: str = arguments.get("agent_id", "")
    message: str = arguments.get("message", "")

    handle = _subagents.get(agent_id)
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
    """Wait for a subagent to complete and return its result."""
    agent_id: str = arguments.get("agent_id", "")

    handle = _subagents.get(agent_id)
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
    """Close a subagent and free its resources."""
    agent_id: str = arguments.get("agent_id", "")

    handle = _subagents.get(agent_id)
    if handle is None:
        return ToolResult(
            output=f"Error: no subagent with id '{agent_id}'",
            is_error=True,
            full_output=f"Error: no subagent with id '{agent_id}'",
        )

    handle.closed = True
    if handle.task and not handle.task.done():
        handle.task.cancel()

    del _subagents[agent_id]
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
