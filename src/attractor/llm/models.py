"""Core data models for the Unified LLM Client.

Defines the provider-agnostic message format, content types, tool definitions,
request/response structures, and streaming event types used across all providers.
"""

from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, enum.Enum):
    """Message roles following the standard LLM conversation model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(str, enum.Enum):
    """Discriminator for ContentPart tagged union."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class StreamEventType(str, enum.Enum):
    """Event types emitted during streaming responses."""

    STREAM_START = "stream_start"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"
    ERROR = "error"


class FinishReason(str, enum.Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    TOOL_USE = "tool_use"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ReasoningEffort(str, enum.Enum):
    """Model reasoning/thinking budget level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Content Parts (Tagged Union)
# ---------------------------------------------------------------------------


@dataclass
class TextContent:
    """Plain text content."""

    kind: ContentKind = field(default=ContentKind.TEXT, init=False)
    text: str = ""


@dataclass
class ImageContent:
    """Image content via URL or base64 data.

    Exactly one of ``url`` or ``base64_data`` must be provided.

    Raises:
        ValueError: If both ``url`` and ``base64_data`` are None.
    """

    kind: ContentKind = field(default=ContentKind.IMAGE, init=False)
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "image/png"

    def __post_init__(self) -> None:
        if self.url is None and self.base64_data is None:
            raise ValueError(
                "ImageContent requires at least one of 'url' or 'base64_data'"
            )


@dataclass
class AudioContent:
    """Audio content."""

    kind: ContentKind = field(default=ContentKind.AUDIO, init=False)
    base64_data: str = ""
    media_type: str = "audio/wav"


@dataclass
class DocumentContent:
    """Document content (PDF, etc.)."""

    kind: ContentKind = field(default=ContentKind.DOCUMENT, init=False)
    base64_data: str = ""
    media_type: str = "application/pdf"


@dataclass
class ToolCallContent:
    """A tool/function call from the assistant.

    Maintains both a parsed ``arguments`` dict and a raw
    ``arguments_json`` string.  ``__post_init__`` syncs whichever
    field was provided so both representations stay consistent.
    """

    kind: ContentKind = field(default=ContentKind.TOOL_CALL, init=False)
    tool_call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_json: str = ""

    def __post_init__(self) -> None:
        """Sync ``arguments`` and ``arguments_json`` if only one is set."""
        if self.arguments_json and not self.arguments:
            try:
                self.arguments = json.loads(self.arguments_json)
            except (json.JSONDecodeError, TypeError):
                pass
        elif self.arguments and not self.arguments_json:
            self.arguments_json = json.dumps(self.arguments)


@dataclass
class ToolResultContent:
    """Result of a tool call execution."""

    kind: ContentKind = field(default=ContentKind.TOOL_RESULT, init=False)
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class ThinkingContent:
    """Model reasoning/thinking content (extended thinking)."""

    kind: ContentKind = field(default=ContentKind.THINKING, init=False)
    text: str = ""


@dataclass
class RedactedThinkingContent:
    """Redacted model reasoning content."""

    kind: ContentKind = field(default=ContentKind.REDACTED_THINKING, init=False)
    data: str = ""


# Union type for all content parts
ContentPart = (
    TextContent
    | ImageContent
    | AudioContent
    | DocumentContent
    | ToolCallContent
    | ToolResultContent
    | ThinkingContent
    | RedactedThinkingContent
)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: list[ContentPart] = field(default_factory=list)

    @staticmethod
    def system(text: str) -> Message:
        """Create a system message.

        Args:
            text: The system instruction text.

        Returns:
            A Message with role SYSTEM and a single TextContent part.
        """
        return Message(role=Role.SYSTEM, content=[TextContent(text=text)])

    @staticmethod
    def user(text: str) -> Message:
        """Create a user message.

        Args:
            text: The user's message text.

        Returns:
            A Message with role USER and a single TextContent part.
        """
        return Message(role=Role.USER, content=[TextContent(text=text)])

    @staticmethod
    def assistant(text: str) -> Message:
        """Create an assistant message.

        Args:
            text: The assistant's response text.

        Returns:
            A Message with role ASSISTANT and a single TextContent part.
        """
        return Message(role=Role.ASSISTANT, content=[TextContent(text=text)])

    @staticmethod
    def tool_result(tool_call_id: str, content: str, is_error: bool = False) -> Message:
        """Create a tool-result message.

        Args:
            tool_call_id: The ID of the tool call this result corresponds to.
            content: The string result (or error description) of execution.
            is_error: Whether the tool execution failed.

        Returns:
            A Message with role TOOL and a single ToolResultContent part.
        """
        return Message(
            role=Role.TOOL,
            content=[
                ToolResultContent(
                    tool_call_id=tool_call_id,
                    content=content,
                    is_error=is_error,
                )
            ],
        )

    def text(self) -> str:
        """Extract concatenated text from all TextContent parts.

        Returns:
            All text parts joined into a single string.
        """
        return "".join(
            part.text for part in self.content if isinstance(part, TextContent)
        )

    def tool_calls(self) -> list[ToolCallContent]:
        """Extract all tool call content parts.

        Returns:
            List of ToolCallContent items in this message.
        """
        return [p for p in self.content if isinstance(p, ToolCallContent)]

    def has_tool_calls(self) -> bool:
        """Check whether this message contains any tool calls.

        Returns:
            True if at least one ToolCallContent part is present.
        """
        return any(isinstance(p, ToolCallContent) for p in self.content)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------


@dataclass
class ToolParameter:
    """A single parameter in a tool's input schema."""

    name: str
    type: str
    description: str = ""
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool that can be invoked by the model."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    strict: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for API submission."""
        return {
            "type": "object",
            "properties": self.parameters.get("properties", {}),
            "required": self.parameters.get("required", []),
        }


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token consumption details."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output).

        Returns:
            Sum of input_tokens and output_tokens.
        """
        return self.input_tokens + self.output_tokens


@dataclass
class RetryPolicy:
    """Configuration for request retry behavior.

    Raises:
        ValueError: If any constraint is violated (negative retries,
            non-positive delay/multiplier, or max_delay < base_delay).
    """

    max_retries: int = 2
    base_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay_seconds <= 0:
            raise ValueError(
                f"base_delay_seconds must be > 0, got {self.base_delay_seconds}"
            )
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier}")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds ({self.max_delay_seconds}) must be "
                f">= base_delay_seconds ({self.base_delay_seconds})"
            )

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate the backoff delay for a given retry attempt.

        Uses exponential backoff: ``base_delay * multiplier^attempt``,
        capped at ``max_delay_seconds``.

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Delay in seconds before the next retry.
        """
        delay = self.base_delay_seconds * (self.multiplier**attempt)
        return min(delay, self.max_delay_seconds)


@dataclass
class Request:
    """A request to an LLM provider."""

    messages: list[Message] = field(default_factory=list)
    model: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    reasoning_effort: ReasoningEffort | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """A response from an LLM provider."""

    message: Message = field(default_factory=lambda: Message(role=Role.ASSISTANT))
    model: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_response_id: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming Events
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """A single event in a streaming response."""

    type: StreamEventType
    text: str = ""
    tool_call: ToolCallContent | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model Catalog Entry
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Metadata about a specific model."""

    model_id: str
    provider: str
    display_name: str = ""
    context_window: int = 128_000
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
