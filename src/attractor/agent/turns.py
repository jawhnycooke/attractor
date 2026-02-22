"""Typed turn records for structured conversation history.

Provides a parallel tracking layer alongside the Message-based history
used for LLM communication. Each Turn captures structured metadata
(timestamps, usage, tool calls) that the flat Message format does not
preserve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import TokenUsage


@dataclass
class UserTurn:
    """A user-submitted input turn.

    Attributes:
        content: The user's input text.
        timestamp: Epoch time when the turn was recorded.
    """

    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantTurn:
    """An LLM assistant response turn.

    Attributes:
        content: The text output from the model.
        tool_calls: Tool invocations requested by the model.
        reasoning: Thinking/reasoning text if available.
        usage: Token consumption for this turn.
        response_id: Provider response identifier.
        timestamp: Epoch time when the turn was recorded.
    """

    content: str
    tool_calls: list[Any] = field(default_factory=list)
    reasoning: str | None = None
    usage: TokenUsage | None = None
    response_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResultsTurn:
    """Results from one or more tool executions.

    Attributes:
        results: One ToolResult per tool call in the round.
        timestamp: Epoch time when the turn was recorded.
    """

    results: list[ToolResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SteeringTurn:
    """A steering message injected between tool rounds.

    Attributes:
        content: The injected steering text.
        timestamp: Epoch time when the turn was recorded.
    """

    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemTurn:
    """An internal system event turn.

    Used for steering messages, config changes, and other internal events
    that are not direct user input or LLM responses.

    Attributes:
        content: The system event content text.
        timestamp: Epoch time when the turn was recorded.
    """

    content: str
    timestamp: float = field(default_factory=time.time)


Turn = UserTurn | AssistantTurn | ToolResultsTurn | SteeringTurn | SystemTurn
