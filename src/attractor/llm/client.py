"""Core LLM Client with provider routing, middleware, and tool loops."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from attractor.llm.errors import ProviderError, SDKError
from attractor.llm.models import (
    FinishReason,
    Message,
    Request,
    Response,
    RetryPolicy,
    StreamEvent,
    ToolCallContent,
    ToolDefinition,
)
from attractor.llm.middleware import Middleware

logger = logging.getLogger(__name__)

# Type alias for tool executors
ToolExecutor = Callable[[ToolCallContent], Coroutine[Any, Any, str]]


class LLMClient:
    """Provider-agnostic LLM client with middleware and tool-loop support.

    Routes requests to the appropriate provider adapter based on model name,
    applies a middleware pipeline, and optionally handles automatic tool
    execution loops with concurrent execution of simultaneous tool calls.
    """

    def __init__(
        self,
        adapters: list[Any] | None = None,
        middleware: list[Middleware] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._adapters = adapters if adapters is not None else self._default_adapters()
        self._middleware: list[Middleware] = middleware or []
        self._retry_policy = retry_policy or RetryPolicy()

    @staticmethod
    def _default_adapters() -> list[Any]:
        """Discover and instantiate available provider adapters.

        Each provider SDK is imported lazily. An ``ImportError`` means the
        package is not installed (expected, logged at DEBUG). Any other
        exception during adapter construction is unexpected and logged at
        WARNING with a full traceback so operators can diagnose the issue.

        Returns:
            List of successfully instantiated provider adapters.
        """
        adapters: list[Any] = []
        try:
            from attractor.llm.adapters.openai_adapter import OpenAIAdapter
        except ImportError:
            logger.debug("openai package not installed; skipping")
        else:
            try:
                adapters.append(OpenAIAdapter())
            except Exception:
                logger.debug("OpenAIAdapter failed to initialize", exc_info=True)
        try:
            from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter
        except ImportError:
            logger.debug("anthropic package not installed; skipping")
        else:
            try:
                adapters.append(AnthropicAdapter())
            except Exception:
                logger.debug("AnthropicAdapter failed to initialize", exc_info=True)
        try:
            from attractor.llm.adapters.gemini_adapter import GeminiAdapter
        except ImportError:
            logger.debug("google-genai package not installed; skipping")
        else:
            try:
                adapters.append(GeminiAdapter())
            except Exception:
                logger.debug("GeminiAdapter failed to initialize", exc_info=True)
        return adapters

    # -----------------------------------------------------------------
    # Provider detection
    # -----------------------------------------------------------------

    def detect_provider(self, model: str) -> Any:
        """Return the first adapter that claims the given model string."""
        for adapter in self._adapters:
            if adapter.detect_model(model):
                return adapter
        raise ValueError(
            f"No provider adapter found for model '{model}'. "
            f"Available adapters: {[a.provider_name() for a in self._adapters]}"
        )

    # -----------------------------------------------------------------
    # Middleware pipeline
    # -----------------------------------------------------------------

    async def _apply_before(self, request: Request) -> Request:
        for mw in self._middleware:
            request = await mw.before_request(request)
        return request

    async def _apply_after(self, response: Response) -> Response:
        for mw in reversed(self._middleware):
            response = await mw.after_response(response)
        return response

    # -----------------------------------------------------------------
    # Retry logic
    # -----------------------------------------------------------------

    async def _complete_with_retry(self, adapter: Any, request: Request) -> Response:
        last_error: Exception | None = None
        for attempt in range(self._retry_policy.max_retries + 1):
            try:
                return await adapter.complete(request)
            except Exception as exc:
                last_error = exc
                # Non-retryable errors should propagate immediately
                if isinstance(exc, SDKError) and not exc.is_retryable:
                    raise
                if attempt < self._retry_policy.max_retries:
                    # Use retry_after from the error if available
                    if isinstance(exc, ProviderError) and exc.retry_after:
                        delay = exc.retry_after
                    else:
                        delay = self._retry_policy.delay_for_attempt(attempt)
                    logger.warning(
                        "LLM request failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._retry_policy.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    # -----------------------------------------------------------------
    # Tool execution helpers
    # -----------------------------------------------------------------

    @staticmethod
    async def _execute_tool(executor: ToolExecutor, tc: ToolCallContent) -> Message:
        """Execute a single tool call and return the result message."""
        try:
            result = await executor(tc)
            return Message.tool_result(tc.tool_call_id, str(result))
        except Exception as exc:
            return Message.tool_result(tc.tool_call_id, str(exc), is_error=True)

    @staticmethod
    async def _execute_tools_concurrently(
        executor: ToolExecutor, tool_calls: list[ToolCallContent]
    ) -> list[Message]:
        """Execute multiple tool calls concurrently with asyncio.gather."""
        tasks = [LLMClient._execute_tool(executor, tc) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))

    # -----------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        """Send a completion request, applying middleware and retries."""
        adapter = self.detect_provider(request.model)
        request = await self._apply_before(request)
        response = await self._complete_with_retry(adapter, request)
        response = await self._apply_after(response)
        return response

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request, applying before-middleware only."""
        adapter = self.detect_provider(request.model)
        request = await self._apply_before(request)
        async for event in adapter.stream(request):
            yield event

    # -----------------------------------------------------------------
    # High-level generate API
    # -----------------------------------------------------------------

    async def generate(
        self,
        prompt: str | list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 10,
        **kwargs: Any,
    ) -> Response:
        """High-level API that handles tool auto-execution loops.

        When the model returns tool calls and a ``tool_executor`` is provided,
        all tool calls in a single round are executed **concurrently** using
        ``asyncio.gather``, and the results are fed back to the model for
        the next round.

        Args:
            prompt: A string (converted to a user message) or list of Messages.
            model: Model identifier string.
            tools: Optional tool definitions to provide to the model.
            tool_executor: Async callable ``(ToolCallContent) -> str`` that
                executes tool calls and returns the result string. Multiple
                simultaneous calls are dispatched concurrently.
            max_tool_rounds: Maximum number of tool-use round trips.
            **kwargs: Additional Request fields (temperature, max_tokens, etc.).
        """
        if isinstance(prompt, str):
            messages = [Message.user(prompt)]
        else:
            messages = list(prompt)

        request = Request(
            messages=messages,
            model=model,
            tools=tools or [],
            **kwargs,
        )

        if max_tool_rounds <= 0:
            return await self.complete(request)

        response: Response | None = None
        for _ in range(max_tool_rounds):
            response = await self.complete(request)

            if response.finish_reason != FinishReason.TOOL_CALLS:
                return response

            tool_calls = response.message.tool_calls()
            if not tool_calls or tool_executor is None:
                return response

            request.messages.append(response.message)

            result_messages = await self._execute_tools_concurrently(
                tool_executor, tool_calls
            )
            request.messages.extend(result_messages)

        assert response is not None
        return response

    async def stream_generate(
        self,
        prompt: str | list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming version of generate (single round, no tool loop)."""
        if isinstance(prompt, str):
            messages = [Message.user(prompt)]
        else:
            messages = list(prompt)

        request = Request(
            messages=messages,
            model=model,
            tools=tools or [],
            **kwargs,
        )

        async for event in self.stream(request):
            yield event

    async def generate_object(
        self,
        prompt: str | list[Message],
        model: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        strict: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a structured JSON object validated against a schema.

        Uses the provider's structured output / response_format capability
        to constrain the model to produce JSON matching the given schema.

        Args:
            prompt: A string or list of Messages.
            model: Model identifier string.
            schema: JSON Schema dict describing the expected output shape.
            schema_name: A name for the schema (used by some providers).
            strict: Whether to enforce strict schema adherence.
            **kwargs: Additional Request fields.

        Returns:
            The parsed JSON object as a Python dict.

        Raises:
            ValueError: If the model output cannot be parsed as valid JSON.
        """
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": strict,
            },
        }

        if isinstance(prompt, str):
            messages = [Message.user(prompt)]
        else:
            messages = list(prompt)

        request = Request(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs,
        )

        response = await self.complete(request)
        text = response.message.text()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {text[:200]!r}") from exc
