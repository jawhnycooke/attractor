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
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    THINKING_START = "thinking_start"
    REASONING_START = "thinking_start"  # spec alias
    REASONING_DELTA = "reasoning_delta"
    THINKING_END = "thinking_end"
    REASONING_END = "thinking_end"  # spec alias
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    PROVIDER_EVENT = "provider_event"
    STEP_FINISH = "step_finish"
    FINISH = "finish"
    ERROR = "error"


@dataclass(frozen=True)
class FinishReason:
    """Why the model stopped generating.

    A dual representation preserving both portable semantics and
    provider-specific detail per unified-llm-spec §3.8.

    Args:
        reason: Unified reason string — one of ``"stop"``, ``"length"``,
            ``"tool_calls"``, ``"content_filter"``, ``"error"``, ``"other"``.
        raw: The provider's native finish reason string (e.g. ``"end_turn"``
            for Anthropic, ``"STOP"`` for Gemini).
    """

    reason: str = "stop"
    raw: str | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FinishReason):
            return self.reason == other.reason
        if isinstance(other, str):
            return self.reason == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.reason)

    def __str__(self) -> str:
        return self.reason

    def __repr__(self) -> str:
        if self.raw:
            return f"FinishReason(reason={self.reason!r}, raw={self.raw!r})"
        return f"FinishReason(reason={self.reason!r})"


# Class-level constants for backward compatibility
FinishReason.STOP = FinishReason("stop")  # type: ignore[attr-defined]
FinishReason.TOOL_CALLS = FinishReason("tool_calls")  # type: ignore[attr-defined]
FinishReason.TOOL_USE = FinishReason("tool_calls")  # type: ignore[attr-defined]  # compat alias
FinishReason.LENGTH = FinishReason("length")  # type: ignore[attr-defined]
FinishReason.CONTENT_FILTER = FinishReason("content_filter")  # type: ignore[attr-defined]
FinishReason.ERROR = FinishReason("error")  # type: ignore[attr-defined]
FinishReason.OTHER = FinishReason("other")  # type: ignore[attr-defined]


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
    detail: str | None = None

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
    """Model reasoning/thinking content (extended thinking).

    The ``signature`` field stores a provider-specific opaque string
    required for round-tripping thinking blocks back to the provider
    (see unified-llm-spec §3.5).
    """

    kind: ContentKind = field(default=ContentKind.THINKING, init=False)
    text: str = ""
    signature: str | None = None


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
    name: str | None = None
    tool_call_id: str | None = None

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

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usage records for aggregation across steps."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


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
class ToolChoice:
    """Controls whether and how the model uses tools."""

    mode: str = "auto"  # auto, none, required, named
    tool_name: str | None = None

    def __post_init__(self) -> None:
        valid_modes = {"auto", "none", "required", "named"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid tool choice mode: {self.mode!r}, must be one of {valid_modes}"
            )
        if self.mode == "named" and not self.tool_name:
            raise ValueError("tool_name is required when mode is 'named'")


@dataclass
class Request:
    """A request to an LLM provider."""

    messages: list[Message] = field(default_factory=list)
    model: str = ""
    provider: str | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: ToolChoice | str | dict[str, Any] | None = None
    system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    reasoning_effort: ReasoningEffort | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] | None = None


@dataclass
class RateLimitInfo:
    """Rate limit metadata from provider response headers."""

    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: float | None = None


@dataclass
class Response:
    """A response from an LLM provider."""

    message: Message = field(default_factory=lambda: Message(role=Role.ASSISTANT))
    model: str = ""
    provider: str = ""
    finish_reason: FinishReason = field(default_factory=lambda: FinishReason.STOP)  # type: ignore[attr-defined]
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_response_id: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    rate_limit: RateLimitInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated text from all text parts in the response message."""
        return self.message.text()

    @property
    def tool_calls(self) -> list[ToolCallContent]:
        """Extracted tool calls from the response message."""
        return self.message.tool_calls()

    @property
    def reasoning(self) -> str | None:
        """Concatenated reasoning/thinking text, or None if absent."""
        parts = [
            p.text
            for p in self.message.content
            if isinstance(p, ThinkingContent) and p.text
        ]
        return "".join(parts) if parts else None


@dataclass
class StepResult:
    """Result of a single step in a multi-step generation."""

    text: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCallContent] = field(default_factory=list)
    tool_results: list[ToolResultContent] = field(default_factory=list)
    finish_reason: FinishReason = field(default_factory=lambda: FinishReason.STOP)  # type: ignore[attr-defined]
    usage: TokenUsage = field(default_factory=TokenUsage)
    response: Response | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of a multi-step generation with tool execution history."""

    text: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCallContent] = field(default_factory=list)
    tool_results: list[ToolResultContent] = field(default_factory=list)
    finish_reason: FinishReason = field(default_factory=lambda: FinishReason.STOP)  # type: ignore[attr-defined]
    usage: TokenUsage = field(default_factory=TokenUsage)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    steps: list[StepResult] = field(default_factory=list)
    response: Response | None = None
    output: Any = None


@dataclass
class TimeoutConfig:
    """Timeout configuration for multi-step operations."""

    total: float | None = None
    per_step: float | None = None


# ---------------------------------------------------------------------------
# Streaming Events
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """A single event in a streaming response."""

    type: StreamEventType
    text: str = ""
    delta: str | None = None
    text_id: str | None = None
    reasoning_delta: str | None = None
    tool_call: ToolCallContent | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None
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
    aliases: list[str] = field(default_factory=list)
