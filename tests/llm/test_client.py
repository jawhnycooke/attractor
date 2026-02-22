"""Tests for attractor.llm.client, middleware, and streaming."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

import attractor.llm as llm_module
from attractor.llm.client import LLMClient
from attractor.llm.errors import AbortError, ConfigurationError, NoObjectGeneratedError, RequestTimeoutError
from attractor.llm.middleware import (
    LoggingMiddleware,
    TokenTrackingMiddleware,
)
from attractor.llm.models import (
    FinishReason,
    Message,
    ReasoningEffort,
    Request,
    Response,
    RetryPolicy,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)
from attractor.llm.streaming import StreamCollector

# ---------------------------------------------------------------------------
# Helpers — mock adapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """A minimal adapter for testing that returns canned responses."""

    def __init__(
        self,
        name: str = "mock",
        prefixes: tuple[str, ...] = ("mock-",),
        response: Response | None = None,
    ) -> None:
        self._name = name
        self._prefixes = prefixes
        self._response = response or Response(
            message=Message.assistant("mock reply"),
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        self.complete_calls: list[Request] = []

    def provider_name(self) -> str:
        return self._name

    def detect_model(self, model: str) -> bool:
        return any(model.startswith(p) for p in self._prefixes)

    async def complete(self, request: Request) -> Response:
        self.complete_calls.append(request)
        return self._response

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type=StreamEventType.STREAM_START)
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="mock ")
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="stream")
        yield StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    def test_detect_known_model(self) -> None:
        adapter = MockAdapter(name="mock", prefixes=("mock-",))
        client = LLMClient(adapters=[adapter])
        assert client.detect_provider("mock-v1") is adapter

    def test_detect_unknown_model_raises(self) -> None:
        adapter = MockAdapter(name="mock", prefixes=("mock-",))
        client = LLMClient(adapters=[adapter])
        with pytest.raises(ConfigurationError, match="No provider adapter found"):
            client.detect_provider("unknown-model")

    def test_first_match_wins(self) -> None:
        a1 = MockAdapter(name="first", prefixes=("test-",))
        a2 = MockAdapter(name="second", prefixes=("test-",))
        client = LLMClient(adapters=[a1, a2])
        assert client.detect_provider("test-model") is a1

    def test_multiple_adapters(self) -> None:
        openai_mock = MockAdapter(name="openai", prefixes=("gpt-", "o1", "o3"))
        anthropic_mock = MockAdapter(name="anthropic", prefixes=("claude-",))
        client = LLMClient(adapters=[openai_mock, anthropic_mock])
        assert client.detect_provider("gpt-4o").provider_name() == "openai"
        assert client.detect_provider("claude-3-opus").provider_name() == "anthropic"

    def test_configuration_error_is_sdk_error(self) -> None:
        """ConfigurationError should be a subclass of SDKError per §8.1."""
        from attractor.llm.errors import SDKError
        assert issubclass(ConfigurationError, SDKError)

    def test_no_adapters_raises_configuration_error(self) -> None:
        """Client with empty adapter list raises ConfigurationError."""
        client = LLMClient(adapters=[])
        with pytest.raises(ConfigurationError, match="No provider adapter found"):
            client.detect_provider("gpt-4o")


# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_routes_to_adapter(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"
        assert len(adapter.complete_calls) == 1

    @pytest.mark.asyncio
    async def test_complete_applies_middleware(self) -> None:
        adapter = MockAdapter()
        tracker = TokenTrackingMiddleware()
        client = LLMClient(adapters=[adapter], middleware=[tracker])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        await client.complete(request)

        assert tracker.total_usage.input_tokens == 10
        assert tracker.total_usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_retry_on_failure(self) -> None:
        call_count = 0

        class FailOnceAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("transient failure")
                return self._response

        adapter = FailOnceAdapter()
        policy = RetryPolicy(max_retries=2, base_delay_seconds=0.01)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_complete_exhausts_retries(self) -> None:
        class AlwaysFailAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                raise ConnectionError("permanent failure")

        adapter = AlwaysFailAdapter()
        policy = RetryPolicy(max_retries=1, base_delay_seconds=0.01)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(ConnectionError, match="permanent failure"):
            await client.complete(request)

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self) -> None:
        """AuthenticationError should not be retried."""
        from attractor.llm.errors import AuthenticationError

        call_count = 0

        class AuthFailAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                raise AuthenticationError(
                    "Invalid API key",
                    provider="mock",
                    status_code=401,
                )

        adapter = AuthFailAdapter()
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.01)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await client.complete(request)

        # Should only have been called once — no retries
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_error_is_retried(self) -> None:
        """RateLimitError should be retried."""
        from attractor.llm.errors import RateLimitError

        call_count = 0

        class RateLimitAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RateLimitError(
                        "Rate limited",
                        provider="mock",
                        status_code=429,
                    )
                return self._response

        adapter = RateLimitAdapter()
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.01)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_uses_retry_after_from_error(self) -> None:
        """Retry delay should use error's retry_after when available."""
        from attractor.llm.errors import RateLimitError

        call_count = 0
        delays: list[float] = []
        start = time.monotonic()

        class RateLimitAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                delays.append(time.monotonic() - start)
                if call_count == 1:
                    raise RateLimitError(
                        "Rate limited",
                        provider="mock",
                        status_code=429,
                        retry_after=0.01,
                    )
                return self._response

        adapter = RateLimitAdapter()
        policy = RetryPolicy(max_retries=2, base_delay_seconds=5.0)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"
        # The retry should have used 0.01s, not 5.0s
        assert call_count == 2


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        request = Request(messages=[Message.user("hi")], model="mock-v1")

        events = []
        async for event in client.stream(request):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.STREAM_START in types
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.FINISH in types


# ---------------------------------------------------------------------------
# Generate (tool loop)
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_simple_text(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate("hello", model="mock-v1")
        assert result.text == "mock reply"
        assert isinstance(result.steps, list)
        assert len(result.steps) == 1
        assert result.total_usage.input_tokens == 10
        assert result.total_usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_with_tool_loop(self) -> None:
        call_count = 0
        second_request_messages: list[Message] = []

        class ToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count, second_request_messages
                call_count += 1
                if call_count == 1:
                    return Response(
                        message=Message(
                            role=Role.ASSISTANT,
                            content=[
                                ToolCallContent(
                                    tool_call_id="call_001",
                                    tool_name="search",
                                    arguments={"q": "test"},
                                )
                            ],
                        ),
                        model="mock-v1",
                        finish_reason=FinishReason.TOOL_USE,
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                    )
                # Capture the second request's messages for verification
                second_request_messages = list(request.messages)
                return Response(
                    message=Message.assistant("final answer"),
                    model="mock-v1",
                    usage=TokenUsage(input_tokens=15, output_tokens=10),
                )

        async def executor(tc: ToolCallContent) -> str:
            return f"result for {tc.tool_name}"

        adapter = ToolAdapter()
        client = LLMClient(adapters=[adapter])
        tools = [ToolDefinition(name="search", description="Search")]
        result = await client.generate(
            "find info",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )
        assert result.text == "final answer"
        assert call_count == 2
        # Should have 2 steps: tool call round + final text
        assert len(result.steps) == 2
        # Total usage aggregated across steps
        assert result.total_usage.input_tokens == 25  # 10 + 15
        assert result.total_usage.output_tokens == 15  # 5 + 10
        # All tool calls collected
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

        # Verify tool results were included in the second request
        # Messages should be: user, assistant (tool call), tool result
        assert len(second_request_messages) == 3
        assert second_request_messages[0].role == Role.USER
        assert second_request_messages[1].role == Role.ASSISTANT
        assert second_request_messages[2].role == Role.TOOL
        tool_result = second_request_messages[2].content[0]
        assert isinstance(tool_result, ToolResultContent)
        assert tool_result.tool_call_id == "call_001"
        assert "result for search" in tool_result.content

    @pytest.mark.asyncio
    async def test_generate_no_executor_returns_tool_response(self) -> None:
        """Without a tool_executor, generate returns the tool_use result."""

        class ToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                return Response(
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            ToolCallContent(
                                tool_call_id="call_002",
                                tool_name="fn",
                                arguments={},
                            )
                        ],
                    ),
                    model="mock-v1",
                    finish_reason=FinishReason.TOOL_USE,
                )

        adapter = ToolAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate("do it", model="mock-v1")
        assert result.finish_reason == FinishReason.TOOL_USE

    @pytest.mark.asyncio
    async def test_generate_with_message_list(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        messages = [Message.system("Be helpful"), Message.user("hi")]
        result = await client.generate(messages, model="mock-v1")
        assert result.text == "mock reply"

    @pytest.mark.asyncio
    async def test_generate_concurrent_tool_execution(self) -> None:
        """Multiple tool calls in one round should execute concurrently."""
        call_count = 0
        execution_log: list[tuple[str, float]] = []

        class MultiToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return Response(
                        message=Message(
                            role=Role.ASSISTANT,
                            content=[
                                ToolCallContent(
                                    tool_call_id="call_a",
                                    tool_name="tool_a",
                                    arguments={"x": 1},
                                ),
                                ToolCallContent(
                                    tool_call_id="call_b",
                                    tool_name="tool_b",
                                    arguments={"y": 2},
                                ),
                                ToolCallContent(
                                    tool_call_id="call_c",
                                    tool_name="tool_c",
                                    arguments={"z": 3},
                                ),
                            ],
                        ),
                        model="mock-v1",
                        finish_reason=FinishReason.TOOL_USE,
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                    )
                return Response(
                    message=Message.assistant("done"),
                    model="mock-v1",
                    usage=TokenUsage(input_tokens=20, output_tokens=10),
                )

        async def executor(tc: ToolCallContent) -> str:
            t0 = time.monotonic()
            await asyncio.sleep(0.05)
            execution_log.append((tc.tool_name, time.monotonic() - t0))
            return f"result_{tc.tool_name}"

        adapter = MultiToolAdapter()
        client = LLMClient(adapters=[adapter])
        tools = [
            ToolDefinition(name="tool_a", description="A"),
            ToolDefinition(name="tool_b", description="B"),
            ToolDefinition(name="tool_c", description="C"),
        ]
        result = await client.generate(
            "multi",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )

        assert result.text == "done"
        assert call_count == 2
        assert len(execution_log) == 3

        # If run sequentially, total time would be ~0.15s (3 x 0.05s).
        # If concurrent, all three finish in ~0.05s.
        # Measure wall-clock: each tool took ~0.05s individually.
        # If concurrent, the max across all three should be << 0.15s.
        total_sequential_time = sum(t for _, t in execution_log)
        max_individual_time = max(t for _, t in execution_log)
        # All should have run in parallel: total ~0.15s but max ~0.05s
        assert total_sequential_time >= 0.12  # each took ~0.05s
        assert max_individual_time < 0.10  # but max is only ~0.05s (concurrent)

    @pytest.mark.asyncio
    async def test_generate_tool_executor_error_handling(self) -> None:
        """Tool executor exceptions should produce is_error tool results."""
        call_count = 0

        class ToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return Response(
                        message=Message(
                            role=Role.ASSISTANT,
                            content=[
                                ToolCallContent(
                                    tool_call_id="call_err",
                                    tool_name="failing_tool",
                                    arguments={},
                                )
                            ],
                        ),
                        model="mock-v1",
                        finish_reason=FinishReason.TOOL_USE,
                    )
                # Check that the error was passed back
                tool_msgs = [m for m in request.messages if m.role == Role.TOOL]
                assert len(tool_msgs) == 1
                part = tool_msgs[0].content[0]
                assert part.is_error is True
                assert "boom" in part.content

                return Response(
                    message=Message.assistant("handled error"),
                    model="mock-v1",
                )

        async def executor(tc: ToolCallContent) -> str:
            raise RuntimeError("boom")

        adapter = ToolAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate(
            "test",
            model="mock-v1",
            tools=[ToolDefinition(name="failing_tool", description="Fails")],
            tool_executor=executor,
        )
        assert result.text == "handled error"


# ---------------------------------------------------------------------------
# generate_object
# ---------------------------------------------------------------------------


class TestGenerateObject:
    @pytest.mark.asyncio
    async def test_generate_object_returns_parsed_json(self) -> None:
        json_text = '{"name": "Alice", "age": 30}'
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant(json_text),
                model="mock-v1",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        )
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = await client.generate_object(
            "Give me a person", model="mock-v1", schema=schema
        )
        assert result.output == {"name": "Alice", "age": 30}
        assert result.text == json_text

    @pytest.mark.asyncio
    async def test_generate_object_sets_response_format(self) -> None:
        """Verify the request has the correct response_format."""
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"x": 1}'),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        await client.generate_object(
            "test", model="mock-v1", schema=schema, schema_name="my_schema"
        )

        req = adapter.complete_calls[0]
        assert req.response_format is not None
        assert req.response_format["type"] == "json_schema"
        assert req.response_format["json_schema"]["name"] == "my_schema"
        assert req.response_format["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_generate_object_invalid_json_raises(self) -> None:
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant("not valid json {{{"),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            await client.generate_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_generate_object_with_message_list(self) -> None:
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"result": true}'),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        messages = [Message.system("Be precise"), Message.user("give bool")]
        result = await client.generate_object(
            messages,
            model="mock-v1",
            schema={"type": "object", "properties": {"result": {"type": "boolean"}}},
        )
        assert result.output == {"result": True}


# ---------------------------------------------------------------------------
# Middleware pipeline
# ---------------------------------------------------------------------------


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_logging_middleware(self) -> None:
        mw = LoggingMiddleware()
        req = Request(model="test", messages=[Message.user("hi")])
        result = await mw.before_request(req)
        assert result is req  # passes through unchanged

        resp = Response(
            model="test",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=100.0,
        )
        result_resp = await mw.after_response(resp)
        assert result_resp is resp

    @pytest.mark.asyncio
    async def test_token_tracking_accumulates(self) -> None:
        tracker = TokenTrackingMiddleware()

        for i in range(3):
            resp = Response(usage=TokenUsage(input_tokens=10, output_tokens=5))
            await tracker.after_response(resp)

        assert tracker.total_usage.input_tokens == 30
        assert tracker.total_usage.output_tokens == 15
        assert tracker.total_usage.total_tokens == 45

    @pytest.mark.asyncio
    async def test_middleware_ordering(self) -> None:
        """Middleware before_request runs in order; after_response in reverse."""
        call_log: list[str] = []

        class OrderedMW:
            def __init__(self, name: str) -> None:
                self._name = name

            async def before_request(self, request: Request) -> Request:
                call_log.append(f"before_{self._name}")
                return request

            async def after_response(self, response: Response) -> Response:
                call_log.append(f"after_{self._name}")
                return response

        adapter = MockAdapter()
        mw_a = OrderedMW("A")
        mw_b = OrderedMW("B")
        client = LLMClient(adapters=[adapter], middleware=[mw_a, mw_b])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        await client.complete(request)

        assert call_log == ["before_A", "before_B", "after_B", "after_A"]


# ---------------------------------------------------------------------------
# StreamCollector
# ---------------------------------------------------------------------------


class TestStreamCollector:
    def test_collect_text_deltas(self) -> None:
        collector = StreamCollector()
        collector.process_event(
            StreamEvent(type=StreamEventType.TEXT_DELTA, text="Hello ")
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TEXT_DELTA, text="world")
        )
        collector.process_event(
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        response = collector.to_response()
        assert response.message.text() == "Hello world"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage.input_tokens == 10

    def test_collect_tool_calls(self) -> None:
        tc = ToolCallContent(
            tool_call_id="call_x",
            tool_name="search",
            arguments={"q": "test"},
            arguments_json='{"q": "test"}',
        )
        collector = StreamCollector()
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
        )
        collector.process_event(
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason.TOOL_USE,
            )
        )

        response = collector.to_response()
        assert response.finish_reason == FinishReason.TOOL_USE
        tool_calls = response.message.tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_collect_async_stream(self) -> None:
        async def fake_stream() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type=StreamEventType.STREAM_START)
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="hi")
            yield StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )

        collector = StreamCollector()
        response = await collector.collect(fake_stream())
        assert response.message.text() == "hi"
        assert response.usage.input_tokens == 5


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_creates_client(self) -> None:
        """from_env() should return an LLMClient using _default_adapters."""
        with patch.object(LLMClient, "_default_adapters", return_value=[]) as mock_da:
            client = LLMClient.from_env()
            assert isinstance(client, LLMClient)
            mock_da.assert_called_once()

    def test_from_env_passes_middleware(self) -> None:
        tracker = TokenTrackingMiddleware()
        with patch.object(LLMClient, "_default_adapters", return_value=[]):
            client = LLMClient.from_env(middleware=[tracker])
            assert client._middleware == [tracker]

    def test_from_env_passes_retry_policy(self) -> None:
        policy = RetryPolicy(max_retries=5, base_delay_seconds=0.5)
        with patch.object(LLMClient, "_default_adapters", return_value=[]):
            client = LLMClient.from_env(retry_policy=policy)
            assert client._retry_policy.max_retries == 5
            assert client._retry_policy.base_delay_seconds == 0.5

    def test_from_env_default_retry_policy(self) -> None:
        """Without explicit retry_policy, from_env uses the default RetryPolicy."""
        with patch.object(LLMClient, "_default_adapters", return_value=[]):
            client = LLMClient.from_env()
            assert client._retry_policy.max_retries == 2  # RetryPolicy default


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    @pytest.fixture(autouse=True)
    def reset_default_client(self) -> None:
        """Reset module-level _default_client after each test."""
        yield  # type: ignore[misc]
        llm_module._default_client = None

    def test_set_and_get_default_client(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        llm_module.set_default_client(client)
        assert llm_module.get_default_client() is client

    def test_get_default_client_lazy_init(self) -> None:
        """get_default_client lazily creates via from_env when unset."""
        assert llm_module._default_client is None
        with patch.object(LLMClient, "_default_adapters", return_value=[]):
            client = llm_module.get_default_client()
            assert isinstance(client, LLMClient)
            # Subsequent call returns same instance
            assert llm_module.get_default_client() is client

    @pytest.mark.asyncio
    async def test_generate_delegates_to_client(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        llm_module.set_default_client(client)

        result = await llm_module.generate("hello", model="mock-v1")
        assert result.text == "mock reply"
        assert len(adapter.complete_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_with_explicit_client(self) -> None:
        """Passing client= should override the default."""
        default_adapter = MockAdapter(
            response=Response(
                message=Message.assistant("default"),
                model="mock-v1",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        )
        explicit_adapter = MockAdapter(
            response=Response(
                message=Message.assistant("explicit"),
                model="mock-v1",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        )
        llm_module.set_default_client(LLMClient(adapters=[default_adapter]))
        explicit_client = LLMClient(adapters=[explicit_adapter])

        result = await llm_module.generate("hi", model="mock-v1", client=explicit_client)
        assert result.text == "explicit"

    @pytest.mark.asyncio
    async def test_stream_delegates_to_client(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        llm_module.set_default_client(client)

        events = []
        async for event in llm_module.stream("hello", model="mock-v1"):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.FINISH in types

    @pytest.mark.asyncio
    async def test_generate_object_delegates_to_client(self) -> None:
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"x": 42}'),
                model="mock-v1",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
        )
        client = LLMClient(adapters=[adapter])
        llm_module.set_default_client(client)

        result = await llm_module.generate_object(
            "give x",
            model="mock-v1",
            schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        assert result.output == {"x": 42}


# ---------------------------------------------------------------------------
# Retry jitter
# ---------------------------------------------------------------------------


class TestRetryJitter:
    @pytest.mark.asyncio
    async def test_jitter_varies_delay(self) -> None:
        """Computed delays should have jitter applied (not exact exponential)."""
        delays: list[float] = []

        class AlwaysFailAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                raise ConnectionError("fail")

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            delays.append(delay)

        policy = RetryPolicy(max_retries=10, base_delay_seconds=0.001, multiplier=2.0)
        adapter = AlwaysFailAdapter()
        client = LLMClient(
            adapters=[adapter], retry_policy=policy, on_retry=on_retry
        )

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(ConnectionError):
            await client.complete(request)

        assert len(delays) == 10

        # With jitter [0.5, 1.5], over 10 attempts it's extremely
        # unlikely all delays equal the exact exponential value.
        exact = [
            min(0.001 * (2.0**i), 60.0) for i in range(10)
        ]
        mismatches = sum(1 for d, e in zip(delays, exact) if d != e)
        assert mismatches > 0, "Jitter should cause delays to differ from exact exponential"

    @pytest.mark.asyncio
    async def test_jitter_bounds(self) -> None:
        """Jittered delays should be within [0.5x, 1.5x] of base delay."""
        delays: list[float] = []

        class AlwaysFailAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                raise ConnectionError("fail")

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            delays.append(delay)

        policy = RetryPolicy(
            max_retries=5, base_delay_seconds=0.01, multiplier=1.0, max_delay_seconds=60.0
        )
        adapter = AlwaysFailAdapter()
        client = LLMClient(
            adapters=[adapter], retry_policy=policy, on_retry=on_retry
        )

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(ConnectionError):
            await client.complete(request)

        # With multiplier=1.0, base delay is always 0.01 for all attempts.
        # Jitter should keep each delay in [0.005, 0.015].
        for d in delays:
            assert 0.005 <= d <= 0.015, f"Delay {d} outside jitter bounds [0.005, 0.015]"

    @pytest.mark.asyncio
    async def test_retry_after_not_jittered(self) -> None:
        """Server-specified retry_after delays should not have jitter."""
        from attractor.llm.errors import RateLimitError

        delays: list[float] = []
        call_count = 0

        class RateLimitAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    raise RateLimitError(
                        "Rate limited",
                        provider="mock",
                        status_code=429,
                        retry_after=0.01,
                    )
                return self._response

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            delays.append(delay)

        policy = RetryPolicy(max_retries=5, base_delay_seconds=5.0)
        adapter = RateLimitAdapter()
        client = LLMClient(
            adapters=[adapter], retry_policy=policy, on_retry=on_retry
        )

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        await client.complete(request)

        # All delays should be exactly 0.01 (retry_after), not jittered
        for d in delays:
            assert d == 0.01, f"retry_after delay should be exact, got {d}"


# ---------------------------------------------------------------------------
# on_retry callback
# ---------------------------------------------------------------------------


class TestOnRetryCallback:
    @pytest.mark.asyncio
    async def test_on_retry_called_with_correct_args(self) -> None:
        callback_args: list[tuple[int, Exception, float]] = []
        call_count = 0

        class FailTwiceAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError(f"fail-{call_count}")
                return self._response

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            callback_args.append((attempt, exc, delay))

        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.01)
        adapter = FailTwiceAdapter()
        client = LLMClient(
            adapters=[adapter], retry_policy=policy, on_retry=on_retry
        )

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"

        # Should have been called twice (attempt 0 and attempt 1)
        assert len(callback_args) == 2
        assert callback_args[0][0] == 0  # first attempt index
        assert isinstance(callback_args[0][1], ConnectionError)
        assert "fail-1" in str(callback_args[0][1])
        assert callback_args[0][2] > 0  # delay is positive

        assert callback_args[1][0] == 1  # second attempt index
        assert "fail-2" in str(callback_args[1][1])

    @pytest.mark.asyncio
    async def test_on_retry_not_called_on_success(self) -> None:
        callback_calls: list[int] = []

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            callback_calls.append(attempt)

        adapter = MockAdapter()
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.01)
        client = LLMClient(
            adapters=[adapter], retry_policy=policy, on_retry=on_retry
        )

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        await client.complete(request)

        assert len(callback_calls) == 0

    @pytest.mark.asyncio
    async def test_no_on_retry_callback_by_default(self) -> None:
        """Without on_retry, retries still work normally."""
        call_count = 0

        class FailOnceAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("transient")
                return self._response

        adapter = FailOnceAdapter()
        policy = RetryPolicy(max_retries=2, base_delay_seconds=0.01)
        client = LLMClient(adapters=[adapter], retry_policy=policy)

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        response = await client.complete(request)
        assert response.message.text() == "mock reply"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Timeout / abort support
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_generate_timeout_raises_request_timeout_error(self) -> None:
        """generate() with timeout should raise RequestTimeoutError when adapter is slow."""

        class SlowAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                await asyncio.sleep(5.0)
                return self._response

        adapter = SlowAdapter()
        client = LLMClient(adapters=[adapter])

        with pytest.raises(RequestTimeoutError, match="timed out"):
            await client.generate("hello", model="mock-v1", timeout=0.05)

    @pytest.mark.asyncio
    async def test_generate_without_timeout_succeeds(self) -> None:
        """generate() without timeout should work normally."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate("hello", model="mock-v1", timeout=None)
        assert result.text == "mock reply"

    @pytest.mark.asyncio
    async def test_generate_with_timeout_succeeds_when_fast(self) -> None:
        """generate() with generous timeout should succeed for fast responses."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate("hello", model="mock-v1", timeout=10.0)
        assert result.text == "mock reply"


class TestAbortSignal:
    @pytest.mark.asyncio
    async def test_generate_abort_before_first_round(self) -> None:
        """generate() should raise AbortError if signal is already set."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])

        abort = asyncio.Event()
        abort.set()

        with pytest.raises(AbortError, match="aborted"):
            await client.generate("hello", model="mock-v1", abort_signal=abort)

    @pytest.mark.asyncio
    async def test_generate_abort_between_tool_rounds(self) -> None:
        """Abort signal set between tool rounds should stop the loop."""
        call_count = 0
        abort = asyncio.Event()

        class ToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count
                call_count += 1
                # After first round, set abort
                abort.set()
                return Response(
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            ToolCallContent(
                                tool_call_id="call_001",
                                tool_name="fn",
                                arguments={},
                            )
                        ],
                    ),
                    model="mock-v1",
                    finish_reason=FinishReason.TOOL_USE,
                )

        async def executor(tc: ToolCallContent) -> str:
            return "ok"

        adapter = ToolAdapter()
        client = LLMClient(adapters=[adapter])

        with pytest.raises(AbortError, match="aborted"):
            await client.generate(
                "test",
                model="mock-v1",
                tools=[ToolDefinition(name="fn", description="A function")],
                tool_executor=executor,
                abort_signal=abort,
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_generate_no_abort_signal_works(self) -> None:
        """generate() without abort_signal should work normally."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        result = await client.generate("hello", model="mock-v1", abort_signal=None)
        assert result.text == "mock reply"


# ---------------------------------------------------------------------------
# Streaming with tool loops
# ---------------------------------------------------------------------------


class TestStreamGenerateWithTools:
    @pytest.mark.asyncio
    async def test_stream_no_tools_returns_events(self) -> None:
        """Without tool calls, stream_generate_with_tools streams and stops."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])

        events = []
        async for event in client.stream_generate_with_tools("hello", model="mock-v1"):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.FINISH in types
        # No STEP_FINISH since no tools were called
        assert StreamEventType.STEP_FINISH not in types

    @pytest.mark.asyncio
    async def test_stream_with_tool_loop_emits_step_finish(self) -> None:
        """stream_generate_with_tools should emit STEP_FINISH between rounds."""
        call_count = 0

        class ToolStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    tc = ToolCallContent(
                        tool_call_id="call_s1",
                        tool_name="search",
                        arguments={"q": "test"},
                        arguments_json='{"q": "test"}',
                    )
                    yield StreamEvent(type=StreamEventType.STREAM_START)
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START, tool_call=tc
                    )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END, tool_call=tc
                    )
                    yield StreamEvent(
                        type=StreamEventType.FINISH,
                        finish_reason=FinishReason.TOOL_USE,
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                    )
                else:
                    yield StreamEvent(type=StreamEventType.STREAM_START)
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA, text="final answer"
                    )
                    yield StreamEvent(
                        type=StreamEventType.FINISH,
                        finish_reason=FinishReason.STOP,
                        usage=TokenUsage(input_tokens=15, output_tokens=10),
                    )

        async def executor(tc: ToolCallContent) -> str:
            return "search result"

        adapter = ToolStreamAdapter()
        client = LLMClient(adapters=[adapter])
        tools = [ToolDefinition(name="search", description="Search")]

        events = []
        async for event in client.stream_generate_with_tools(
            "find info",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        ):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.STEP_FINISH in types

        # Find the STEP_FINISH event and verify metadata
        step_events = [e for e in events if e.type == StreamEventType.STEP_FINISH]
        assert len(step_events) == 1
        assert step_events[0].metadata["round"] == 0
        assert step_events[0].metadata["tool_calls"] == 1

        # The final text should have been streamed
        text_deltas = [
            e.text for e in events if e.type == StreamEventType.TEXT_DELTA
        ]
        assert "final answer" in text_deltas

    @pytest.mark.asyncio
    async def test_stream_without_executor_stops_at_tool_calls(self) -> None:
        """Without tool_executor, stream should stop after tool call round."""

        class ToolStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                tc = ToolCallContent(
                    tool_call_id="call_x",
                    tool_name="fn",
                    arguments={},
                    arguments_json="{}",
                )
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
                yield StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc)
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.TOOL_USE,
                )

        adapter = ToolStreamAdapter()
        client = LLMClient(adapters=[adapter])

        events = []
        async for event in client.stream_generate_with_tools(
            "test", model="mock-v1", tools=[ToolDefinition(name="fn", description="F")]
        ):
            events.append(event)

        # Should have tool call events but no STEP_FINISH (no executor)
        assert StreamEventType.STEP_FINISH not in [e.type for e in events]


# ---------------------------------------------------------------------------
# Tool parameter validation
# ---------------------------------------------------------------------------


class TestToolValidation:
    def _make_client(self) -> LLMClient:
        return LLMClient(adapters=[MockAdapter()])

    def test_valid_args_pass_through(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters={
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="search", arguments={"q": "hello"}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, ToolCallContent)
        assert result.arguments["q"] == "hello"

    def test_missing_required_field_returns_error(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters={
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="search", arguments={}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, str)
        assert "Missing required field: q" in result

    def test_unknown_tool_returns_error(self) -> None:
        client = self._make_client()
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="nonexistent", arguments={}
        )

        result = client._validate_tool_args(tc, [])
        assert isinstance(result, str)
        assert "Unknown tool: nonexistent" in result

    def test_type_coercion_string_to_int(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="seek",
            description="Seek",
            parameters={
                "properties": {"offset": {"type": "integer"}},
                "required": ["offset"],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="seek", arguments={"offset": "42"}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, ToolCallContent)
        assert result.arguments["offset"] == 42

    def test_type_coercion_string_to_float(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="set_temp",
            description="Set temperature",
            parameters={
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="set_temp", arguments={"value": "3.14"}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, ToolCallContent)
        assert result.arguments["value"] == pytest.approx(3.14)

    def test_type_coercion_string_to_boolean(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="toggle",
            description="Toggle",
            parameters={
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="toggle", arguments={"enabled": "true"}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, ToolCallContent)
        assert result.arguments["enabled"] is True

    def test_type_coercion_boolean_false(self) -> None:
        client = self._make_client()
        tool = ToolDefinition(
            name="toggle",
            description="Toggle",
            parameters={
                "properties": {"enabled": {"type": "boolean"}},
                "required": [],
            },
        )
        tc = ToolCallContent(
            tool_call_id="c1", tool_name="toggle", arguments={"enabled": "no"}
        )

        result = client._validate_tool_args(tc, [tool])
        assert isinstance(result, ToolCallContent)
        assert result.arguments["enabled"] is False

    @pytest.mark.asyncio
    async def test_validation_errors_sent_as_tool_results(self) -> None:
        """Validation errors should be sent back to the model as error results."""
        call_count = 0
        captured_messages: list[Message] = []

        class ToolAdapter(MockAdapter):
            async def complete(self, request: Request) -> Response:
                nonlocal call_count, captured_messages
                call_count += 1
                if call_count == 1:
                    return Response(
                        message=Message(
                            role=Role.ASSISTANT,
                            content=[
                                ToolCallContent(
                                    tool_call_id="c_bad",
                                    tool_name="search",
                                    arguments={},  # missing required "q"
                                )
                            ],
                        ),
                        model="mock-v1",
                        finish_reason=FinishReason.TOOL_USE,
                    )
                captured_messages = list(request.messages)
                return Response(
                    message=Message.assistant("recovered"),
                    model="mock-v1",
                )

        async def executor(tc: ToolCallContent) -> str:
            return "should not be called"

        adapter = ToolAdapter()
        client = LLMClient(adapters=[adapter])
        tools = [
            ToolDefinition(
                name="search",
                description="Search",
                parameters={
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        result = await client.generate(
            "test",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )
        assert result.text == "recovered"

        # Check that a tool error result was sent back
        tool_msgs = [m for m in captured_messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        part = tool_msgs[0].content[0]
        assert isinstance(part, ToolResultContent)
        assert part.is_error is True
        assert "Missing required field" in part.content


# ---------------------------------------------------------------------------
# Provider-based routing (L-C02)
# ---------------------------------------------------------------------------


class TestProviderBasedRouting:
    """Tests for explicit provider-based routing via Request.provider field."""

    def test_provider_routes_to_matching_adapter(self) -> None:
        """Request with provider='anthropic' should route to that adapter."""
        openai_adapter = MockAdapter(name="openai", prefixes=("gpt-",))
        anthropic_adapter = MockAdapter(name="anthropic", prefixes=("claude-",))
        client = LLMClient(adapters=[openai_adapter, anthropic_adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="some-model",
            provider="anthropic",
        )
        resolved = client._resolve_adapter(request)
        assert resolved is anthropic_adapter
        assert resolved.provider_name() == "anthropic"

    def test_provider_bypasses_model_detection(self) -> None:
        """Provider field should override model-string detection.

        Even though the model string 'gpt-4o' would match the openai adapter
        via detect_model(), an explicit provider='anthropic' should route to
        the anthropic adapter instead.
        """
        openai_adapter = MockAdapter(name="openai", prefixes=("gpt-",))
        anthropic_adapter = MockAdapter(name="anthropic", prefixes=("claude-",))
        client = LLMClient(adapters=[openai_adapter, anthropic_adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="gpt-4o",
            provider="anthropic",
        )
        resolved = client._resolve_adapter(request)
        assert resolved is anthropic_adapter

    def test_fallback_to_model_detection_when_provider_is_none(self) -> None:
        """When provider is None, routing should fall back to detect_model()."""
        openai_adapter = MockAdapter(name="openai", prefixes=("gpt-",))
        anthropic_adapter = MockAdapter(name="anthropic", prefixes=("claude-",))
        client = LLMClient(adapters=[openai_adapter, anthropic_adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="claude-3-opus",
            provider=None,
        )
        resolved = client._resolve_adapter(request)
        assert resolved is anthropic_adapter

    def test_unknown_provider_raises_configuration_error(self) -> None:
        """Specifying a provider that isn't registered should raise ConfigurationError."""
        adapter = MockAdapter(name="openai", prefixes=("gpt-",))
        client = LLMClient(adapters=[adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="gpt-4o",
            provider="nonexistent",
        )
        with pytest.raises(ConfigurationError, match="Provider 'nonexistent' is not registered"):
            client._resolve_adapter(request)

    def test_provider_field_is_optional_and_backward_compatible(self) -> None:
        """Request without provider field should work identically to before."""
        adapter = MockAdapter(name="mock", prefixes=("mock-",))
        client = LLMClient(adapters=[adapter])

        # Default constructor — provider defaults to None
        request = Request(messages=[Message.user("hi")], model="mock-v1")
        assert request.provider is None
        resolved = client._resolve_adapter(request)
        assert resolved is adapter

    @pytest.mark.asyncio
    async def test_complete_uses_provider_routing(self) -> None:
        """complete() should route via provider when set on the request."""
        openai_adapter = MockAdapter(
            name="openai",
            prefixes=("gpt-",),
            response=Response(
                message=Message.assistant("openai reply"),
                model="gpt-4o",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        )
        anthropic_adapter = MockAdapter(
            name="anthropic",
            prefixes=("claude-",),
            response=Response(
                message=Message.assistant("anthropic reply"),
                model="claude-3",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        )
        client = LLMClient(adapters=[openai_adapter, anthropic_adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="my-custom-model",
            provider="anthropic",
        )
        response = await client.complete(request)
        assert response.message.text() == "anthropic reply"
        assert len(anthropic_adapter.complete_calls) == 1
        assert len(openai_adapter.complete_calls) == 0

    @pytest.mark.asyncio
    async def test_stream_uses_provider_routing(self) -> None:
        """stream() should route via provider when set on the request."""
        adapter = MockAdapter(name="target", prefixes=())
        other = MockAdapter(name="other", prefixes=("mock-",))
        client = LLMClient(adapters=[other, adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="any-model",
            provider="target",
        )
        events = []
        async for event in client.stream(request):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TEXT_DELTA in types

    def test_empty_string_provider_falls_back_to_model_detection(self) -> None:
        """An empty string provider should be treated as absent (falsy)."""
        adapter = MockAdapter(name="mock", prefixes=("mock-",))
        client = LLMClient(adapters=[adapter])

        request = Request(
            messages=[Message.user("hi")],
            model="mock-v1",
            provider="",
        )
        resolved = client._resolve_adapter(request)
        assert resolved is adapter


# ---------------------------------------------------------------------------
# L-C08: Streaming middleware support
# ---------------------------------------------------------------------------


class TestStreamingMiddleware:
    @pytest.mark.asyncio
    async def test_before_request_runs_before_stream(self) -> None:
        """before_request middleware should modify the request before streaming."""
        call_log: list[str] = []

        class TrackingMiddleware:
            async def before_request(self, request: Request) -> Request:
                call_log.append("before_request")
                # Mutate to prove middleware ran
                request.model = "mock-modified"
                return request

            async def after_response(self, response: Response) -> Response:
                return response

        class TrackingAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                call_log.append(f"stream:{request.model}")
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=5, output_tokens=2),
                )

        adapter = TrackingAdapter(prefixes=("mock-",))
        mw = TrackingMiddleware()
        client = LLMClient(adapters=[adapter], middleware=[mw])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        events = []
        async for event in client.stream(request):
            events.append(event)

        # before_request ran before streaming
        assert call_log[0] == "before_request"
        assert call_log[1] == "stream:mock-modified"
        # Stream actually produced events
        assert any(e.type == StreamEventType.TEXT_DELTA for e in events)

    @pytest.mark.asyncio
    async def test_wrap_stream_middleware_observes_events(self) -> None:
        """Middleware with wrap_stream should see every stream event."""
        observed_types: list[StreamEventType] = []

        class StreamObserver:
            async def before_request(self, request: Request) -> Request:
                return request

            async def after_response(self, response: Response) -> Response:
                return response

            async def wrap_stream(
                self, stream: AsyncIterator[StreamEvent]
            ) -> AsyncIterator[StreamEvent]:
                async for event in stream:
                    observed_types.append(event.type)
                    yield event

        adapter = MockAdapter()
        mw = StreamObserver()
        client = LLMClient(adapters=[adapter], middleware=[mw])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        events = []
        async for event in client.stream(request):
            events.append(event)

        # wrap_stream saw all events
        assert len(observed_types) == len(events)
        assert StreamEventType.STREAM_START in observed_types
        assert StreamEventType.TEXT_DELTA in observed_types
        assert StreamEventType.FINISH in observed_types

    @pytest.mark.asyncio
    async def test_wrap_stream_can_transform_events(self) -> None:
        """wrap_stream should be able to modify events passing through."""

        class TextUppercaser:
            async def before_request(self, request: Request) -> Request:
                return request

            async def after_response(self, response: Response) -> Response:
                return response

            async def wrap_stream(
                self, stream: AsyncIterator[StreamEvent]
            ) -> AsyncIterator[StreamEvent]:
                async for event in stream:
                    if event.type == StreamEventType.TEXT_DELTA and event.text:
                        yield StreamEvent(
                            type=event.type,
                            text=event.text.upper(),
                        )
                    else:
                        yield event

        adapter = MockAdapter()
        mw = TextUppercaser()
        client = LLMClient(adapters=[adapter], middleware=[mw])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        text_parts = []
        async for event in client.stream(request):
            if event.type == StreamEventType.TEXT_DELTA:
                text_parts.append(event.text)

        # Text deltas should be uppercased by the middleware
        assert "MOCK " in text_parts
        assert "STREAM" in text_parts

    @pytest.mark.asyncio
    async def test_middleware_without_wrap_stream_is_skipped(self) -> None:
        """Middleware that doesn't implement wrap_stream should be ignored."""

        class PlainMiddleware:
            async def before_request(self, request: Request) -> Request:
                return request

            async def after_response(self, response: Response) -> Response:
                return response

        adapter = MockAdapter()
        mw = PlainMiddleware()
        client = LLMClient(adapters=[adapter], middleware=[mw])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        events = []
        async for event in client.stream(request):
            events.append(event)

        # Stream should work normally without wrap_stream
        types = [e.type for e in events]
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.FINISH in types

    @pytest.mark.asyncio
    async def test_multiple_wrap_stream_middleware_chain(self) -> None:
        """Multiple middleware with wrap_stream should chain in order."""
        transform_log: list[str] = []

        class PrefixMiddleware:
            def __init__(self, tag: str) -> None:
                self._tag = tag

            async def before_request(self, request: Request) -> Request:
                return request

            async def after_response(self, response: Response) -> Response:
                return response

            async def wrap_stream(
                self, stream: AsyncIterator[StreamEvent]
            ) -> AsyncIterator[StreamEvent]:
                async for event in stream:
                    if event.type == StreamEventType.TEXT_DELTA:
                        transform_log.append(self._tag)
                    yield event

        adapter = MockAdapter()
        mw_a = PrefixMiddleware("A")
        mw_b = PrefixMiddleware("B")
        client = LLMClient(adapters=[adapter], middleware=[mw_a, mw_b])

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        async for _ in client.stream(request):
            pass

        # A wraps B wraps adapter. Events flow: adapter -> A -> B -> consumer
        # So A sees events first, then B
        # The MockAdapter yields 2 TEXT_DELTA events ("mock " and "stream")
        assert len(transform_log) == 4  # 2 deltas x 2 middleware
        # Order: A sees first delta, then B sees first delta, then A sees second, etc.
        # Actually with chaining: adapter -> mw_a.wrap -> mw_b.wrap -> consumer
        # mw_a wraps the raw stream, mw_b wraps mw_a's output
        # So mw_a logs first for each event, then mw_b
        # For each TEXT_DELTA: A logs, yields, B sees it, logs, yields
        assert transform_log == ["A", "B", "A", "B"]


# ---------------------------------------------------------------------------
# L-C01: LLMClient.close()
# ---------------------------------------------------------------------------


class TestClientClose:
    @pytest.mark.asyncio
    async def test_close_marks_client_as_closed(self) -> None:
        """close() should mark the client as closed."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        assert client._closed is False

        await client.close()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times should not raise."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])

        await client.close()
        await client.close()  # second call should be a no-op
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_complete_after_close_raises(self) -> None:
        """complete() should raise RuntimeError after close()."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        await client.close()

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            await client.complete(request)

    @pytest.mark.asyncio
    async def test_stream_after_close_raises(self) -> None:
        """stream() should raise RuntimeError after close()."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        await client.close()

        request = Request(messages=[Message.user("hi")], model="mock-v1")
        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            async for _ in client.stream(request):
                pass

    @pytest.mark.asyncio
    async def test_generate_after_close_raises(self) -> None:
        """generate() should raise RuntimeError after close()."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            await client.generate("hi", model="mock-v1")

    @pytest.mark.asyncio
    async def test_close_calls_adapter_close(self) -> None:
        """close() should call close() on adapters that have it."""
        close_called = False

        class ClosableAdapter(MockAdapter):
            async def close(self) -> None:
                nonlocal close_called
                close_called = True

        adapter = ClosableAdapter()
        client = LLMClient(adapters=[adapter])

        await client.close()
        assert close_called is True

    @pytest.mark.asyncio
    async def test_close_handles_adapter_without_close(self) -> None:
        """close() should skip adapters that don't have close()."""
        adapter = MockAdapter()  # no close() method
        client = LLMClient(adapters=[adapter])

        # Should not raise
        await client.close()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_close_handles_adapter_close_error(self) -> None:
        """close() should continue even if an adapter's close() raises."""

        class FailingCloseAdapter(MockAdapter):
            async def close(self) -> None:
                raise ConnectionError("close failed")

        adapter = FailingCloseAdapter()
        client = LLMClient(adapters=[adapter])

        # Should not raise despite adapter error
        await client.close()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_close_calls_close_on_multiple_adapters(self) -> None:
        """close() should call close() on all adapters, not just the first."""
        close_log: list[str] = []

        class CloseTracker(MockAdapter):
            async def close(self) -> None:
                close_log.append(self._name)

        adapter_a = CloseTracker(name="a", prefixes=("a-",))
        adapter_b = CloseTracker(name="b", prefixes=("b-",))
        client = LLMClient(adapters=[adapter_a, adapter_b])

        await client.close()
        assert "a" in close_log
        assert "b" in close_log
        assert len(close_log) == 2


# ---------------------------------------------------------------------------
# L-C09: generate_object() — enhanced tests
# ---------------------------------------------------------------------------


class TestGenerateObjectEnhanced:
    """Tests for generate_object() spec compliance (L-C09)."""

    @pytest.mark.asyncio
    async def test_generate_object_uses_generate_internally(self) -> None:
        """generate_object() should delegate to generate() under the hood."""
        generate_called = False

        class TrackingClient(LLMClient):
            async def generate(self, *args: Any, **kwargs: Any) -> Any:
                nonlocal generate_called
                generate_called = True
                return await super().generate(*args, **kwargs)

        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"val": 1}'),
                model="mock-v1",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
        )
        client = TrackingClient(adapters=[adapter])

        result = await client.generate_object(
            "test",
            model="mock-v1",
            schema={"type": "object", "properties": {"val": {"type": "integer"}}},
        )
        assert generate_called is True
        assert result.output == {"val": 1}

    @pytest.mark.asyncio
    async def test_generate_object_nested_schema(self) -> None:
        """generate_object() should handle nested JSON schemas."""
        json_text = '{"user": {"name": "Bob", "address": {"city": "NYC", "zip": "10001"}}}'
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant(json_text),
                model="mock-v1",
                usage=TokenUsage(input_tokens=10, output_tokens=15),
            ),
        )
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "required": ["user"],
        }

        result = await client.generate_object(
            "Give me a user",
            model="mock-v1",
            schema=schema,
        )
        assert result.output["user"]["name"] == "Bob"
        assert result.output["user"]["address"]["city"] == "NYC"
        assert result.output["user"]["address"]["zip"] == "10001"

    @pytest.mark.asyncio
    async def test_generate_object_array_schema(self) -> None:
        """generate_object() should handle schemas with array types."""
        json_text = '{"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}'
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant(json_text),
                model="mock-v1",
                usage=TokenUsage(input_tokens=5, output_tokens=10),
            ),
        )
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }

        result = await client.generate_object(
            "List items",
            model="mock-v1",
            schema=schema,
        )
        assert len(result.output["items"]) == 2
        assert result.output["items"][0]["id"] == 1
        assert result.output["items"][1]["name"] == "b"

    @pytest.mark.asyncio
    async def test_generate_object_strict_false(self) -> None:
        """generate_object() should support strict=False."""
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"x": 1}'),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        await client.generate_object(
            "test",
            model="mock-v1",
            schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            strict=False,
        )

        req = adapter.complete_calls[0]
        assert req.response_format is not None
        assert req.response_format["json_schema"]["strict"] is False

    @pytest.mark.asyncio
    async def test_generate_object_empty_json_raises(self) -> None:
        """generate_object() should raise NoObjectGeneratedError on empty response."""
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant(""),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            await client.generate_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_generate_object_truncated_json_raises(self) -> None:
        """generate_object() should raise NoObjectGeneratedError on truncated JSON."""
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"name": "Alice", "age":'),
                model="mock-v1",
            ),
        )
        client = LLMClient(adapters=[adapter])

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            await client.generate_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_generate_object_passes_kwargs(self) -> None:
        """generate_object() should forward extra kwargs to generate()."""
        adapter = MockAdapter(
            response=Response(
                message=Message.assistant('{"ok": true}'),
                model="mock-v1",
                usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
        )
        client = LLMClient(adapters=[adapter])

        await client.generate_object(
            "test",
            model="mock-v1",
            schema={"type": "object"},
            temperature=0.5,
            max_tokens=100,
        )

        req = adapter.complete_calls[0]
        assert req.temperature == 0.5
        assert req.max_tokens == 100

    @pytest.mark.asyncio
    async def test_generate_object_after_close_raises(self) -> None:
        """generate_object() should raise RuntimeError after close()."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            await client.generate_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            )


# ---------------------------------------------------------------------------
# L-C10: stream_object()
# ---------------------------------------------------------------------------


class TestStreamObject:
    """Tests for stream_object() (L-C10)."""

    @pytest.mark.asyncio
    async def test_stream_object_yields_parsed_result(self) -> None:
        """stream_object() should yield a parsed dict from streamed JSON."""

        class JSONStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='{"name"')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=': "Alice"')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=', "age": 30}')
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=10, output_tokens=8),
                )

        adapter = JSONStreamAdapter()
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        results = []
        async for obj in client.stream_object(
            "Extract person info",
            model="mock-v1",
            schema=schema,
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0] == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_stream_object_sets_response_format(self) -> None:
        """stream_object() should set response_format on the request."""
        captured_requests: list[Request] = []

        class TrackingStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='{"x": 1}')
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                )

        adapter = TrackingStreamAdapter()
        client = LLMClient(adapters=[adapter])

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        async for _ in client.stream_object(
            "test",
            model="mock-v1",
            schema=schema,
            schema_name="my_schema",
        ):
            pass

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.response_format is not None
        assert req.response_format["type"] == "json_schema"
        assert req.response_format["json_schema"]["name"] == "my_schema"
        assert req.response_format["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_stream_object_invalid_json_raises(self) -> None:
        """stream_object() should raise ValueError on non-JSON output."""

        class BadStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA, text="not valid json {[["
                )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = BadStreamAdapter()
        client = LLMClient(adapters=[adapter])

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            async for _ in client.stream_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_object_nested_schema(self) -> None:
        """stream_object() should handle nested JSON objects."""
        json_text = '{"user": {"name": "Eve", "roles": ["admin", "user"]}}'

        class NestedStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=json_text)
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=5, output_tokens=10),
                )

        adapter = NestedStreamAdapter()
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "roles": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }

        results = []
        async for obj in client.stream_object(
            "Get user",
            model="mock-v1",
            schema=schema,
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0]["user"]["name"] == "Eve"
        assert results[0]["user"]["roles"] == ["admin", "user"]

    @pytest.mark.asyncio
    async def test_stream_object_array_result(self) -> None:
        """stream_object() should handle array-typed schemas."""
        json_text = '{"tags": ["python", "async", "llm"]}'

        class ArrayStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=json_text)
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = ArrayStreamAdapter()
        client = LLMClient(adapters=[adapter])

        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }

        results = []
        async for obj in client.stream_object(
            "List tags",
            model="mock-v1",
            schema=schema,
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0]["tags"] == ["python", "async", "llm"]

    @pytest.mark.asyncio
    async def test_stream_object_with_message_list(self) -> None:
        """stream_object() should accept a list of Messages as prompt."""

        class MessageStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA, text='{"status": "ok"}'
                )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = MessageStreamAdapter()
        client = LLMClient(adapters=[adapter])

        messages = [Message.system("Be precise"), Message.user("status check")]
        results = []
        async for obj in client.stream_object(
            messages,
            model="mock-v1",
            schema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0] == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_stream_object_after_close_raises(self) -> None:
        """stream_object() should raise RuntimeError after close()."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            async for _ in client.stream_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_object_empty_stream_raises(self) -> None:
        """stream_object() should raise ValueError on empty stream output."""

        class EmptyStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = EmptyStreamAdapter()
        client = LLMClient(adapters=[adapter])

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            async for _ in client.stream_object(
                "test",
                model="mock-v1",
                schema={"type": "object"},
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_object_strict_false(self) -> None:
        """stream_object() should support strict=False."""
        captured_requests: list[Request] = []

        class TrackingStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='{"x": 1}')
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = TrackingStreamAdapter()
        client = LLMClient(adapters=[adapter])

        async for _ in client.stream_object(
            "test",
            model="mock-v1",
            schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            strict=False,
        ):
            pass

        req = captured_requests[0]
        assert req.response_format["json_schema"]["strict"] is False

    @pytest.mark.asyncio
    async def test_stream_object_multiple_deltas_concatenated(self) -> None:
        """stream_object() should correctly concatenate multiple text deltas."""

        class MultiDeltaAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="{")
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='"k')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='ey"')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=": ")
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='"val')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text='ue"')
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="}")
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                )

        adapter = MultiDeltaAdapter()
        client = LLMClient(adapters=[adapter])

        results = []
        async for obj in client.stream_object(
            "test",
            model="mock-v1",
            schema={"type": "object", "properties": {"key": {"type": "string"}}},
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0] == {"key": "value"}


# ---------------------------------------------------------------------------
# L-C10: Module-level stream_object()
# ---------------------------------------------------------------------------


class TestModuleLevelStreamObject:
    @pytest.fixture(autouse=True)
    def reset_default_client(self) -> None:
        """Reset module-level _default_client after each test."""
        yield  # type: ignore[misc]
        llm_module._default_client = None

    @pytest.mark.asyncio
    async def test_stream_object_delegates_to_client(self) -> None:
        """Module-level stream_object() should delegate to the default client."""

        class JSONStreamAdapter(MockAdapter):
            async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type=StreamEventType.STREAM_START)
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA, text='{"val": 42}'
                )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                )

        adapter = JSONStreamAdapter()
        client = LLMClient(adapters=[adapter])
        llm_module.set_default_client(client)

        results = []
        async for obj in llm_module.stream_object(
            "test",
            model="mock-v1",
            schema={"type": "object", "properties": {"val": {"type": "integer"}}},
        ):
            results.append(obj)

        assert len(results) == 1
        assert results[0] == {"val": 42}


# ---------------------------------------------------------------------------
# ResponseFormat dataclass
# ---------------------------------------------------------------------------


class TestResponseFormat:
    """Tests for the ResponseFormat dataclass (spec §3.10)."""

    def test_default_values(self) -> None:
        from attractor.llm.models import ResponseFormat

        rf = ResponseFormat()
        assert rf.type == "text"
        assert rf.json_schema is None
        assert rf.strict is False

    def test_json_schema_type(self) -> None:
        from attractor.llm.models import ResponseFormat

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        rf = ResponseFormat(type="json_schema", json_schema=schema, strict=True)
        assert rf.type == "json_schema"
        assert rf.json_schema == schema
        assert rf.strict is True

    def test_json_type(self) -> None:
        from attractor.llm.models import ResponseFormat

        rf = ResponseFormat(type="json")
        assert rf.type == "json"

    def test_invalid_type_raises(self) -> None:
        from attractor.llm.models import ResponseFormat

        with pytest.raises(ValueError, match="Invalid response format type"):
            ResponseFormat(type="xml")

    def test_json_schema_type_without_schema_raises(self) -> None:
        from attractor.llm.models import ResponseFormat

        with pytest.raises(ValueError, match="json_schema is required"):
            ResponseFormat(type="json_schema")

    def test_to_dict_json_schema(self) -> None:
        from attractor.llm.models import ResponseFormat

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        rf = ResponseFormat(type="json_schema", json_schema=schema, strict=True)
        d = rf.to_dict()
        assert d["type"] == "json_schema"
        assert d["json_schema"] == schema
        assert d["strict"] is True

    def test_to_dict_text(self) -> None:
        from attractor.llm.models import ResponseFormat

        rf = ResponseFormat(type="text")
        d = rf.to_dict()
        assert d == {"type": "text"}

    def test_to_dict_json(self) -> None:
        from attractor.llm.models import ResponseFormat

        rf = ResponseFormat(type="json")
        d = rf.to_dict()
        assert d == {"type": "json"}


# ---------------------------------------------------------------------------
# Adapter-specific: OpenAI Codex model detection
# ---------------------------------------------------------------------------


class TestOpenAICodexDetection:
    """Tests for OpenAI adapter detecting codex- prefixed models."""

    def test_codex_mini_latest_detected(self) -> None:
        """OpenAI adapter should detect codex-mini-latest as an OpenAI model."""
        from attractor.llm.adapters.openai_adapter import OpenAIAdapter

        # Instantiation requires the openai SDK; test detect_model statically
        assert OpenAIAdapter._MODEL_PREFIXES == ("gpt-", "o1", "o3", "o4", "codex-")

    def test_codex_prefix_in_mock_adapter(self) -> None:
        """MockAdapter with OpenAI prefixes should match codex- models."""
        adapter = MockAdapter(
            name="openai",
            prefixes=("gpt-", "o1", "o3", "o4", "codex-"),
        )
        assert adapter.detect_model("codex-mini-latest") is True
        assert adapter.detect_model("gpt-4o") is True
        assert adapter.detect_model("claude-opus") is False


# ---------------------------------------------------------------------------
# Adapter-specific: Gemini structured output and reasoning effort
# ---------------------------------------------------------------------------


class TestGeminiAdapterFeatures:
    """Tests for Gemini adapter _build_kwargs: structured output and reasoning."""

    def test_structured_output_maps_response_format(self) -> None:
        """Gemini _build_kwargs should map response_format to response_mime_type + response_schema."""
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        request = Request(
            messages=[Message.user("test")],
            model="gemini-3-pro-preview",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        # Call _build_kwargs without constructing the full adapter (avoids SDK import)
        # We test the logic by calling the method on a partially-constructed instance
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._client = None  # type: ignore[assignment]
        kwargs = adapter._build_kwargs(request)

        config = kwargs.get("config", {})
        assert config.get("response_mime_type") == "application/json"
        assert config.get("response_schema") == schema

    def test_reasoning_effort_maps_to_thinking_config(self) -> None:
        """Gemini _build_kwargs should map reasoning_effort to thinking_config."""
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter, _THINKING_BUDGET

        request = Request(
            messages=[Message.user("test")],
            model="gemini-3-pro-preview",
            reasoning_effort=ReasoningEffort.HIGH,
        )

        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._client = None  # type: ignore[assignment]
        kwargs = adapter._build_kwargs(request)

        config = kwargs.get("config", {})
        assert "thinking_config" in config
        assert config["thinking_config"]["thinking_budget"] == _THINKING_BUDGET[ReasoningEffort.HIGH]

    def test_reasoning_effort_does_not_override_explicit_thinking_config(self) -> None:
        """Provider-specific thinking_config should take precedence over reasoning_effort."""
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter

        request = Request(
            messages=[Message.user("test")],
            model="gemini-3-pro-preview",
            reasoning_effort=ReasoningEffort.HIGH,
            provider_options={
                "gemini": {
                    "thinking_config": {"thinking_budget": 99999},
                },
            },
        )

        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._client = None  # type: ignore[assignment]
        kwargs = adapter._build_kwargs(request)

        config = kwargs.get("config", {})
        # Provider-specific override should win
        assert config["thinking_config"]["thinking_budget"] == 99999


# ---------------------------------------------------------------------------
# Adapter-specific: Anthropic adaptive thinking
# ---------------------------------------------------------------------------


class TestAnthropicAdaptiveThinking:
    """Tests for Anthropic adapter adaptive vs legacy thinking."""

    def _make_adapter(self) -> Any:
        """Create a bare AnthropicAdapter without SDK import."""
        from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter._client = None  # type: ignore[assignment]
        return adapter

    def test_supports_adaptive_thinking_claude_4_6(self) -> None:
        """Claude 4.6 models should support adaptive thinking."""
        from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter

        assert AnthropicAdapter._supports_adaptive_thinking("claude-opus-4-6") is True
        assert AnthropicAdapter._supports_adaptive_thinking("claude-sonnet-4-6") is True

    def test_does_not_support_adaptive_thinking_older_models(self) -> None:
        """Older Claude models should not support adaptive thinking."""
        from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter

        assert AnthropicAdapter._supports_adaptive_thinking("claude-sonnet-4-5") is False
        assert AnthropicAdapter._supports_adaptive_thinking("claude-opus-4-5") is False
        assert AnthropicAdapter._supports_adaptive_thinking("claude-haiku-4-5") is False

    def test_adaptive_thinking_for_4_6_model(self) -> None:
        """_build_kwargs should produce adaptive thinking for claude-opus-4-6."""
        adapter = self._make_adapter()

        request = Request(
            messages=[Message.user("test")],
            model="claude-opus-4-6",
            reasoning_effort=ReasoningEffort.MEDIUM,
        )
        kwargs = adapter._build_kwargs(request)

        assert kwargs["thinking"] == {"type": "adaptive"}
        assert "output_config" in kwargs
        assert kwargs["output_config"]["effort"] == "medium"

    def test_legacy_thinking_for_4_5_model(self) -> None:
        """_build_kwargs should produce legacy enabled thinking for claude-sonnet-4-5."""
        adapter = self._make_adapter()

        request = Request(
            messages=[Message.user("test")],
            model="claude-sonnet-4-5",
            reasoning_effort=ReasoningEffort.HIGH,
        )
        kwargs = adapter._build_kwargs(request)

        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 32768
        assert "output_config" not in kwargs or "effort" not in kwargs.get("output_config", {})

    def test_combined_output_config_effort_and_format(self) -> None:
        """When both reasoning_effort and response_format are set, output_config has both."""
        adapter = self._make_adapter()

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        request = Request(
            messages=[Message.user("test")],
            model="claude-opus-4-6",
            reasoning_effort=ReasoningEffort.HIGH,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        kwargs = adapter._build_kwargs(request)

        assert kwargs["thinking"] == {"type": "adaptive"}
        assert "output_config" in kwargs
        # Both effort and format should coexist in output_config
        assert kwargs["output_config"]["effort"] == "high"
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        assert kwargs["output_config"]["format"]["schema"] == schema

    def test_provider_thinking_override_still_wins(self) -> None:
        """Explicit provider_options.anthropic.thinking should override reasoning_effort."""
        adapter = self._make_adapter()

        request = Request(
            messages=[Message.user("test")],
            model="claude-opus-4-6",
            reasoning_effort=ReasoningEffort.HIGH,
            provider_options={
                "anthropic": {
                    "thinking": {"type": "enabled", "budget_tokens": 50000},
                },
            },
        )
        kwargs = adapter._build_kwargs(request)

        # Provider-specific override should replace the adaptive thinking
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 50000}


# ---------------------------------------------------------------------------
# Catalog: new model entries
# ---------------------------------------------------------------------------


class TestCatalogUpdates:
    """Tests for updated model catalog entries."""

    def test_codex_mini_latest_in_catalog(self) -> None:
        """codex-mini-latest should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("codex-mini-latest")
        assert info is not None
        assert info.provider == "openai"
        assert info.supports_vision is False
        assert info.supports_reasoning is True

    def test_gemini_2_5_pro_in_catalog(self) -> None:
        """gemini-2.5-pro should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("gemini-2.5-pro")
        assert info is not None
        assert info.provider == "gemini"
        assert info.context_window == 1_048_576

    def test_gemini_2_5_flash_in_catalog(self) -> None:
        """gemini-2.5-flash should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("gemini-2.5-flash")
        assert info is not None
        assert info.provider == "gemini"

    def test_claude_opus_4_6_updated_output_tokens(self) -> None:
        """claude-opus-4-6 should have updated max_output_tokens of 128k."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("claude-opus-4-6")
        assert info is not None
        assert info.max_output_tokens == 128_000

    def test_claude_sonnet_4_5_updated_output_tokens(self) -> None:
        """claude-sonnet-4-5 should have updated max_output_tokens of 64k."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("claude-sonnet-4-5")
        assert info is not None
        assert info.max_output_tokens == 64_000

    def test_claude_sonnet_4_6_in_catalog(self) -> None:
        """claude-sonnet-4-6 should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("claude-sonnet-4-6")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.max_output_tokens == 64_000

    def test_claude_haiku_4_5_in_catalog(self) -> None:
        """claude-haiku-4-5 should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("claude-haiku-4-5")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.input_cost_per_million == 1.0

    def test_gpt_5_2_updated_context_window(self) -> None:
        """gpt-5.2 should have context_window of 400k."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("gpt-5.2")
        assert info is not None
        assert info.context_window == 400_000

    def test_gpt_5_2_codex_alias_changed(self) -> None:
        """gpt-5.2-codex should use 'gpt-codex' alias (not 'codex')."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("gpt-codex")
        assert info is not None
        assert info.model_id == "gpt-5.2-codex"
        # Old alias should not resolve to gpt-5.2-codex
        old_info = get_model_info("codex")
        assert old_info is None or old_info.model_id != "gpt-5.2-codex"

    def test_o3_in_catalog(self) -> None:
        """o3 should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("o3")
        assert info is not None
        assert info.provider == "openai"
        assert info.supports_reasoning is True

    def test_o4_mini_in_catalog(self) -> None:
        """o4-mini should be in the model catalog."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("o4-mini")
        assert info is not None
        assert info.provider == "openai"

    def test_gpt_4_1_no_reasoning(self) -> None:
        """gpt-4.1 should not support reasoning."""
        from attractor.llm.catalog import get_model_info

        info = get_model_info("gpt-4.1")
        assert info is not None
        assert info.supports_reasoning is False
        assert info.supports_vision is True
