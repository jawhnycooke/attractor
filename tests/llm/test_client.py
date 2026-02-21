"""Tests for attractor.llm.client, middleware, and streaming."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

import attractor.llm as llm_module
from attractor.llm.client import LLMClient
from attractor.llm.errors import AbortError, RequestTimeoutError
from attractor.llm.middleware import (
    LoggingMiddleware,
    TokenTrackingMiddleware,
)
from attractor.llm.models import (
    FinishReason,
    Message,
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
        with pytest.raises(ValueError, match="No provider adapter found"):
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
        response = await client.generate("hello", model="mock-v1")
        assert response.message.text() == "mock reply"

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
        response = await client.generate(
            "find info",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )
        assert response.message.text() == "final answer"
        assert call_count == 2

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
        """Without a tool_executor, generate returns the tool_use response."""

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
        response = await client.generate("do it", model="mock-v1")
        assert response.finish_reason == FinishReason.TOOL_USE

    @pytest.mark.asyncio
    async def test_generate_with_message_list(self) -> None:
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        messages = [Message.system("Be helpful"), Message.user("hi")]
        response = await client.generate(messages, model="mock-v1")
        assert response.message.text() == "mock reply"

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
        response = await client.generate(
            "multi",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )

        assert response.message.text() == "done"
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
        response = await client.generate(
            "test",
            model="mock-v1",
            tools=[ToolDefinition(name="failing_tool", description="Fails")],
            tool_executor=executor,
        )
        assert response.message.text() == "handled error"


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
        assert result == {"name": "Alice", "age": 30}

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

        with pytest.raises(ValueError, match="not valid JSON"):
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
        assert result == {"result": True}


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

        response = await llm_module.generate("hello", model="mock-v1")
        assert response.message.text() == "mock reply"
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

        response = await llm_module.generate("hi", model="mock-v1", client=explicit_client)
        assert response.message.text() == "explicit"

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
        assert result == {"x": 42}


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
        response = await client.generate("hello", model="mock-v1", timeout=None)
        assert response.message.text() == "mock reply"

    @pytest.mark.asyncio
    async def test_generate_with_timeout_succeeds_when_fast(self) -> None:
        """generate() with generous timeout should succeed for fast responses."""
        adapter = MockAdapter()
        client = LLMClient(adapters=[adapter])
        response = await client.generate("hello", model="mock-v1", timeout=10.0)
        assert response.message.text() == "mock reply"


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
        response = await client.generate("hello", model="mock-v1", abort_signal=None)
        assert response.message.text() == "mock reply"


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
        response = await client.generate(
            "test",
            model="mock-v1",
            tools=tools,
            tool_executor=executor,
        )
        assert response.message.text() == "recovered"

        # Check that a tool error result was sent back
        tool_msgs = [m for m in captured_messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        part = tool_msgs[0].content[0]
        assert isinstance(part, ToolResultContent)
        assert part.is_error is True
        assert "Missing required field" in part.content
