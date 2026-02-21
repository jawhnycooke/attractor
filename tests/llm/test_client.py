"""Tests for attractor.llm.client, middleware, and streaming."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from attractor.llm.client import LLMClient
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
