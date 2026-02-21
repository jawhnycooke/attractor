"""Tool registry — central dispatch for agent tool calls.

The registry maps tool names to handler callables and their definitions.
Unknown tool names produce an error result rather than raising exceptions,
so the agent loop can continue gracefully.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.llm.models import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Outcome of a single tool execution.

    Attributes:
        output: The text sent back to the LLM (possibly truncated).
        is_error: True if the tool produced an error.
        full_output: The untruncated original output.
    """

    output: str
    is_error: bool = False
    full_output: str = ""


# Type alias for tool handler functions.
ToolHandler = Callable[..., Awaitable[ToolResult]]


class ToolRegistry:
    """Maps tool names to handlers and definitions.

    Example::

        registry = ToolRegistry()
        registry.register("read_file", read_file_handler, read_file_def)
        result = await registry.dispatch("read_file", {"path": "foo.py"}, env)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        definition: ToolDefinition,
    ) -> None:
        """Register a tool handler and its definition.

        Args:
            name: The tool name used for dispatch.
            handler: Async callable implementing the tool.
            definition: JSON-schema definition sent to the LLM.
        """
        self._handlers[name] = handler
        self._definitions[name] = definition

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        environment: ExecutionEnvironment,
    ) -> ToolResult:
        """Look up and execute a tool by name.

        Returns an error ToolResult for unknown tools rather than raising.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                output=f"Error: unknown tool '{name}'",
                is_error=True,
                full_output=f"Error: unknown tool '{name}'",
            )
        try:
            return await handler(arguments=arguments, environment=environment)
        except Exception as exc:
            logger.exception("Unhandled exception in tool '%s'", name)
            error_msg = f"Error executing tool '{name}': {exc}"
            return ToolResult(
                output=error_msg,
                is_error=True,
                full_output=error_msg,
            )

    def definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._definitions.values())

    def has_tool(self, name: str) -> bool:
        """Return True if a tool with *name* is registered."""
        return name in self._handlers

    def tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._handlers.keys())
