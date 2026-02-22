"""Unified LLM Client - Provider-agnostic LLM interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from attractor.llm.errors import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    InvalidToolCallError,
    NetworkError,
    NoObjectGeneratedError,
    NotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    SDKError,
    ServerError,
    StreamError,
)
from attractor.llm.models import (
    ContentPart,
    GenerateResult,
    ImageContent,
    Message,
    RateLimitInfo,
    Request,
    Response,
    ResponseFormat,
    Role,
    StepResult,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    TokenUsage,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
)
from attractor.llm.catalog import MODELS, get_latest_model, get_model_info, list_models
from attractor.llm.client import LLMClient, ToolExecutor
from attractor.llm.streaming import StreamResult

# ---------------------------------------------------------------------------
# Module-level default client (spec §2.5)
# ---------------------------------------------------------------------------

_default_client: LLMClient | None = None


def set_default_client(client: LLMClient) -> None:
    """Set the module-level default client."""
    global _default_client
    _default_client = client


def get_default_client() -> LLMClient:
    """Get the module-level default client, lazily creating from env."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient.from_env()
    return _default_client


async def generate(
    prompt: str | list[Message],
    model: str,
    tools: list[ToolDefinition] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_tool_rounds: int = 10,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Module-level generate using the default client."""
    c = client or get_default_client()
    return await c.generate(
        prompt,
        model,
        tools=tools,
        tool_executor=tool_executor,
        max_tool_rounds=max_tool_rounds,
        **kwargs,
    )


async def stream(
    prompt: str | list[Message],
    model: str,
    tools: list[ToolDefinition] | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Module-level stream using the default client."""
    c = client or get_default_client()
    async for event in c.stream_generate(prompt, model, tools=tools, **kwargs):
        yield event


async def generate_object(
    prompt: str | list[Message],
    model: str,
    schema: dict[str, Any],
    schema_name: str = "response",
    strict: bool = True,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Module-level generate_object using the default client."""
    c = client or get_default_client()
    return await c.generate_object(
        prompt,
        model,
        schema=schema,
        schema_name=schema_name,
        strict=strict,
        **kwargs,
    )


async def stream_object(
    prompt: str | list[Message],
    model: str,
    schema: dict[str, Any],
    schema_name: str = "response",
    strict: bool = True,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Module-level stream_object using the default client."""
    c = client or get_default_client()
    async for obj in c.stream_object(
        prompt,
        model,
        schema=schema,
        schema_name=schema_name,
        strict=strict,
        **kwargs,
    ):
        yield obj


__all__ = [
    "LLMClient",
    "ToolExecutor",
    # Module-level API
    "set_default_client",
    "get_default_client",
    "generate",
    "stream",
    "generate_object",
    "stream_object",
    # Streaming
    "StreamResult",
    # Catalog
    "MODELS",
    "get_model_info",
    "list_models",
    "get_latest_model",
    # Errors
    "AbortError",
    "AccessDeniedError",
    "AuthenticationError",
    "ConfigurationError",
    "ContentFilterError",
    "ContextLengthError",
    "InvalidRequestError",
    "InvalidToolCallError",
    "NetworkError",
    "NoObjectGeneratedError",
    "NotFoundError",
    "ProviderError",
    "QuotaExceededError",
    "RateLimitError",
    "RequestTimeoutError",
    "SDKError",
    "ServerError",
    "StreamError",
    # Models
    "ContentPart",
    "GenerateResult",
    "ImageContent",
    "Message",
    "RateLimitInfo",
    "Request",
    "Response",
    "ResponseFormat",
    "Role",
    "StepResult",
    "StreamEvent",
    "StreamEventType",
    "TextContent",
    "ThinkingContent",
    "TokenUsage",
    "ToolCallContent",
    "ToolDefinition",
    "ToolResultContent",
]
