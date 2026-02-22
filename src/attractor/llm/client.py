"""Core LLM Client with provider routing, middleware, and tool loops."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from attractor.llm.errors import AbortError, ProviderError, RequestTimeoutError, SDKError
from attractor.llm.models import (
    FinishReason,
    GenerateResult,
    Message,
    Request,
    Response,
    RetryPolicy,
    StepResult,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
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
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> None:
        self._adapters = adapters if adapters is not None else self._default_adapters()
        self._middleware: list[Middleware] = middleware or []
        self._retry_policy = retry_policy or RetryPolicy()
        self._on_retry = on_retry

    @classmethod
    def from_env(
        cls,
        middleware: list[Middleware] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> LLMClient:
        """Create a client from environment variables.

        Reads API keys from standard environment variables and registers
        only providers whose keys are present. This is the recommended
        setup for most applications.

        Environment variables:
            OPENAI_API_KEY: OpenAI API key
            ANTHROPIC_API_KEY: Anthropic API key
            GEMINI_API_KEY or GOOGLE_API_KEY: Google Gemini API key
        """
        return cls(
            adapters=cls._default_adapters(),
            middleware=middleware,
            retry_policy=retry_policy,
        )

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
                    # Use retry_after from the error if available;
                    # server-specified delay is respected as-is (no jitter).
                    if isinstance(exc, ProviderError) and exc.retry_after:
                        delay = exc.retry_after
                    else:
                        delay = self._retry_policy.delay_for_attempt(attempt)
                        # Apply jitter to computed delays only
                        delay *= random.uniform(0.5, 1.5)
                    if self._on_retry:
                        self._on_retry(attempt, exc, delay)
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
    # Tool argument validation
    # -----------------------------------------------------------------

    def _validate_tool_args(
        self,
        tc: ToolCallContent,
        tools: list[ToolDefinition],
    ) -> ToolCallContent | str:
        """Validate tool call arguments against schema.

        Returns the (possibly repaired) ToolCallContent if valid,
        or an error string if validation fails.
        """
        tool_def = next((t for t in tools if t.name == tc.tool_name), None)
        if tool_def is None:
            return f"Unknown tool: {tc.tool_name}"

        schema = tool_def.to_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        args = dict(tc.arguments)  # copy

        # Check required fields
        for req_field in required:
            if req_field not in args:
                return f"Missing required field: {req_field}"

        # Basic type coercion
        for key, value in list(args.items()):
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type == "integer" and isinstance(value, str):
                    try:
                        args[key] = int(value)
                    except ValueError:
                        pass
                elif expected_type == "number" and isinstance(value, str):
                    try:
                        args[key] = float(value)
                    except ValueError:
                        pass
                elif expected_type == "boolean" and isinstance(value, str):
                    args[key] = value.lower() in ("true", "1", "yes")

        # Return updated tool call
        tc.arguments = args
        tc.arguments_json = json.dumps(args)
        return tc

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
        timeout: float | None = None,
        abort_signal: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        """High-level API that handles tool auto-execution loops.

        When the model returns tool calls and a ``tool_executor`` is provided,
        all tool calls in a single round are executed **concurrently** using
        ``asyncio.gather``, and the results are fed back to the model for
        the next round.

        Returns a :class:`GenerateResult` that includes per-step history
        and aggregated token usage in addition to the final response.

        Args:
            prompt: A string (converted to a user message) or list of Messages.
            model: Model identifier string.
            tools: Optional tool definitions to provide to the model.
            tool_executor: Async callable ``(ToolCallContent) -> str`` that
                executes tool calls and returns the result string. Multiple
                simultaneous calls are dispatched concurrently.
            max_tool_rounds: Maximum number of tool-use round trips.
            timeout: Per-round timeout in seconds. If a single complete()
                call exceeds this, raises RequestTimeoutError.
            abort_signal: An asyncio.Event that, when set, aborts the
                generation loop with an AbortError.
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
            response = await self.complete(request)
            step = self._build_step(response, [])
            return self._build_result(response, [step])

        steps: list[StepResult] = []
        response: Response | None = None
        for _ in range(max_tool_rounds):
            # Check abort signal before each round
            if abort_signal and abort_signal.is_set():
                raise AbortError("Generation aborted")

            # Apply timeout to the complete() call
            try:
                if timeout:
                    response = await asyncio.wait_for(
                        self.complete(request), timeout=timeout
                    )
                else:
                    response = await self.complete(request)
            except asyncio.TimeoutError:
                raise RequestTimeoutError(
                    f"Generation timed out after {timeout}s",
                    provider="",
                    retryable=True,
                )

            if response.finish_reason != FinishReason.TOOL_CALLS:
                steps.append(self._build_step(response, []))
                return self._build_result(response, steps)

            tool_calls = response.message.tool_calls()
            if not tool_calls or tool_executor is None:
                steps.append(self._build_step(response, []))
                return self._build_result(response, steps)

            request.messages.append(response.message)

            # Collect tool results for step tracking
            round_tool_results: list[ToolResultContent] = []

            # Validate tool arguments before execution
            if request.tools:
                validated_calls: list[ToolCallContent] = []
                validation_errors: list[tuple[ToolCallContent, str]] = []
                for tc in tool_calls:
                    result = self._validate_tool_args(tc, request.tools)
                    if isinstance(result, str):
                        validation_errors.append((tc, result))
                    else:
                        validated_calls.append(result)

                # Send validation errors back as tool results
                for tc, error in validation_errors:
                    error_msg = Message.tool_result(tc.tool_call_id, error, is_error=True)
                    request.messages.append(error_msg)
                    round_tool_results.extend(
                        p for p in error_msg.content if isinstance(p, ToolResultContent)
                    )

                # Execute validated calls
                if validated_calls:
                    result_messages = await self._execute_tools_concurrently(
                        tool_executor, validated_calls
                    )
                    request.messages.extend(result_messages)
                    for msg in result_messages:
                        round_tool_results.extend(
                            p for p in msg.content if isinstance(p, ToolResultContent)
                        )
            else:
                result_messages = await self._execute_tools_concurrently(
                    tool_executor, tool_calls
                )
                request.messages.extend(result_messages)
                for msg in result_messages:
                    round_tool_results.extend(
                        p for p in msg.content if isinstance(p, ToolResultContent)
                    )

            steps.append(self._build_step(response, round_tool_results))

        assert response is not None
        steps.append(self._build_step(response, []))
        return self._build_result(response, steps)

    @staticmethod
    def _build_step(
        response: Response, tool_results: list[ToolResultContent]
    ) -> StepResult:
        """Build a StepResult from a response and its tool results."""
        return StepResult(
            text=response.message.text() or "",
            reasoning=response.reasoning,
            tool_calls=response.message.tool_calls(),
            tool_results=tool_results,
            finish_reason=response.finish_reason or FinishReason.STOP,
            usage=response.usage or TokenUsage(),
            response=response,
        )

    @staticmethod
    def _build_result(
        response: Response, steps: list[StepResult]
    ) -> GenerateResult:
        """Build a GenerateResult from the final response and all steps."""
        total_usage = TokenUsage()
        all_tool_calls: list[ToolCallContent] = []
        all_tool_results: list[ToolResultContent] = []
        for step in steps:
            total_usage = total_usage + step.usage
            all_tool_calls.extend(step.tool_calls)
            all_tool_results.extend(step.tool_results)

        return GenerateResult(
            text=response.message.text() or "",
            reasoning=response.reasoning,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            finish_reason=response.finish_reason or FinishReason.STOP,
            usage=response.usage or TokenUsage(),
            total_usage=total_usage,
            steps=steps,
            response=response,
        )

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

    async def stream_generate_with_tools(
        self,
        prompt: str | list[Message],
        model: str,
        tools: list[ToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming generation with automatic tool execution loops.

        Streams the model's response. When tool calls are detected,
        executes them concurrently, emits a STEP_FINISH event,
        and starts a new streaming round with tool results.
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

        from attractor.llm.streaming import StreamCollector

        for round_num in range(max_tool_rounds + 1):
            collector = StreamCollector()

            async for event in self.stream(request):
                collector.process_event(event)
                yield event

            step_response = collector.to_response()

            # If no tool calls or no executor, we're done
            if step_response.finish_reason != FinishReason.TOOL_CALLS:
                return

            tool_calls = step_response.message.tool_calls()
            if not tool_calls or tool_executor is None:
                return

            # Execute tools concurrently
            result_messages = await self._execute_tools_concurrently(
                tool_executor, tool_calls
            )

            # Emit step_finish event
            yield StreamEvent(
                type=StreamEventType.STEP_FINISH,
                usage=step_response.usage,
                metadata={"round": round_num, "tool_calls": len(tool_calls)},
            )

            # Append to conversation for next round
            request.messages.append(step_response.message)
            request.messages.extend(result_messages)

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
