"""Middleware system for the Unified LLM Client.

Middleware can intercept and transform requests before they reach a provider
and responses after they return, enabling cross-cutting concerns like
logging, token tracking, and retry logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from attractor.llm.models import Request, Response, RetryPolicy, TokenUsage

logger = logging.getLogger(__name__)


@runtime_checkable
class Middleware(Protocol):
    """Protocol for request/response middleware."""

    async def before_request(self, request: Request) -> Request:
        """Transform or inspect a request before it is sent to a provider.

        Args:
            request: The outgoing LLM request.

        Returns:
            The (potentially modified) request to forward downstream.
        """
        ...

    async def after_response(self, response: Response) -> Response:
        """Transform or inspect a response after it is received from a provider.

        Args:
            response: The incoming LLM response.

        Returns:
            The (potentially modified) response to return upstream.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in Middleware
# ---------------------------------------------------------------------------


class LoggingMiddleware:
    """Logs request metadata and response latency/token usage."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._level = log_level

    async def before_request(self, request: Request) -> Request:
        msg_count = len(request.messages)
        tool_count = len(request.tools)
        logger.log(
            self._level,
            "LLM request: model=%s messages=%d tools=%d",
            request.model,
            msg_count,
            tool_count,
        )
        return request

    async def after_response(self, response: Response) -> Response:
        logger.log(
            self._level,
            "LLM response: model=%s finish=%s tokens=%d latency=%.0fms",
            response.model,
            response.finish_reason.value,
            response.usage.total_tokens,
            response.latency_ms,
        )
        return response


@dataclass
class TokenTrackingMiddleware:
    """Accumulates token usage across all calls."""

    total_usage: TokenUsage = field(default_factory=TokenUsage)

    async def before_request(self, request: Request) -> Request:
        return request

    async def after_response(self, response: Response) -> Response:
        self.total_usage.input_tokens += response.usage.input_tokens
        self.total_usage.output_tokens += response.usage.output_tokens
        self.total_usage.reasoning_tokens += response.usage.reasoning_tokens
        self.total_usage.cache_read_tokens += response.usage.cache_read_tokens
        self.total_usage.cache_write_tokens += response.usage.cache_write_tokens
        return response


class RetryMiddleware:
    """Wraps calls with retry logic and exponential backoff.

    Note: This middleware only provides timing/logging around retries.
    The actual retry loop lives in LLMClient since middleware cannot
    re-invoke the completion call itself.
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self.policy = policy or RetryPolicy()
        self.last_attempt: int = 0

    async def before_request(self, request: Request) -> Request:
        return request

    async def after_response(self, response: Response) -> Response:
        self.last_attempt = 0
        return response
