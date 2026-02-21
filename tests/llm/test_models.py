"""Tests for attractor.llm.models."""

from __future__ import annotations


from attractor.llm.models import (
    ContentKind,
    FinishReason,
    ImageContent,
    Message,
    Request,
    Response,
    RetryPolicy,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Message creation helpers
# ---------------------------------------------------------------------------


class TestMessageFactories:
    def test_system_message(self) -> None:
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.text() == "You are helpful."
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)

    def test_user_message(self) -> None:
        msg = Message.user("Hello")
        assert msg.role == Role.USER
        assert msg.text() == "Hello"

    def test_assistant_message(self) -> None:
        msg = Message.assistant("Hi there")
        assert msg.role == Role.ASSISTANT
        assert msg.text() == "Hi there"

    def test_tool_result_message(self) -> None:
        msg = Message.tool_result("call_123", "result data")
        assert msg.role == Role.TOOL
        assert len(msg.content) == 1
        part = msg.content[0]
        assert isinstance(part, ToolResultContent)
        assert part.tool_call_id == "call_123"
        assert part.content == "result data"
        assert part.is_error is False

    def test_tool_result_error(self) -> None:
        msg = Message.tool_result("call_456", "something broke", is_error=True)
        part = msg.content[0]
        assert isinstance(part, ToolResultContent)
        assert part.is_error is True

    def test_text_concatenation(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="Hello "), TextContent(text="world")],
        )
        assert msg.text() == "Hello world"

    def test_text_ignores_non_text_parts(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                TextContent(text="result: "),
                ToolCallContent(tool_name="fn", arguments={}),
            ],
        )
        assert msg.text() == "result: "

    def test_message_name_field(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content=[TextContent(text="data")],
            name="my_tool",
        )
        assert msg.name == "my_tool"

    def test_message_tool_call_id_field(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content=[TextContent(text="output")],
            tool_call_id="call_abc",
        )
        assert msg.tool_call_id == "call_abc"

    def test_message_name_defaults_to_none(self) -> None:
        msg = Message.user("test")
        assert msg.name is None
        assert msg.tool_call_id is None


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


class TestContentParts:
    def test_text_content_kind(self) -> None:
        tc = TextContent(text="hello")
        assert tc.kind == ContentKind.TEXT

    def test_image_content_kind(self) -> None:
        img = ImageContent(url="https://example.com/img.png")
        assert img.kind == ContentKind.IMAGE
        assert img.media_type == "image/png"

    def test_tool_call_content_auto_id(self) -> None:
        tc = ToolCallContent(tool_name="search", arguments={"q": "test"})
        assert tc.tool_call_id.startswith("call_")
        assert tc.kind == ContentKind.TOOL_CALL

    def test_thinking_content(self) -> None:
        t = ThinkingContent(text="Let me think...")
        assert t.kind == ContentKind.THINKING

    def test_thinking_content_signature(self) -> None:
        t = ThinkingContent(text="reasoning", signature="sig_abc123")
        assert t.signature == "sig_abc123"

    def test_thinking_content_signature_default_none(self) -> None:
        t = ThinkingContent(text="reasoning")
        assert t.signature is None


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:
    def test_has_tool_calls_true(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content=[ToolCallContent(tool_name="fn", arguments={})],
        )
        assert msg.has_tool_calls() is True

    def test_has_tool_calls_false(self) -> None:
        msg = Message.assistant("no tools here")
        assert msg.has_tool_calls() is False

    def test_tool_calls_extraction(self) -> None:
        tc1 = ToolCallContent(tool_name="fn_a", arguments={"x": 1})
        tc2 = ToolCallContent(tool_name="fn_b", arguments={"y": 2})
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextContent(text="planning"), tc1, tc2],
        )
        calls = msg.tool_calls()
        assert len(calls) == 2
        assert calls[0].tool_name == "fn_a"
        assert calls[1].tool_name == "fn_b"


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_tokens(self) -> None:
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_defaults_zero(self) -> None:
        u = TokenUsage()
        assert u.total_tokens == 0
        assert u.reasoning_tokens == 0
        assert u.cache_read_tokens == 0

    def test_total_excludes_reasoning(self) -> None:
        u = TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=30)
        assert u.total_tokens == 150  # reasoning not added to total


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_delay_exponential(self) -> None:
        policy = RetryPolicy(base_delay_seconds=1.0, multiplier=2.0)
        assert policy.delay_for_attempt(0) == 1.0
        assert policy.delay_for_attempt(1) == 2.0
        assert policy.delay_for_attempt(2) == 4.0
        assert policy.delay_for_attempt(3) == 8.0

    def test_delay_capped(self) -> None:
        policy = RetryPolicy(
            base_delay_seconds=1.0, multiplier=10.0, max_delay_seconds=30.0
        )
        assert policy.delay_for_attempt(0) == 1.0
        assert policy.delay_for_attempt(1) == 10.0
        assert policy.delay_for_attempt(2) == 30.0  # capped
        assert policy.delay_for_attempt(10) == 30.0  # still capped

    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.base_delay_seconds == 1.0


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_to_json_schema(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        schema = tool.to_json_schema()
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_empty_parameters(self) -> None:
        tool = ToolDefinition(name="noop", description="Does nothing")
        schema = tool.to_json_schema()
        assert schema["properties"] == {}
        assert schema["required"] == []


# ---------------------------------------------------------------------------
# FinishReason
# ---------------------------------------------------------------------------


class TestFinishReason:
    def test_equality_by_reason(self) -> None:
        a = FinishReason("stop", raw="end_turn")
        b = FinishReason("stop", raw="stop_sequence")
        assert a == b

    def test_inequality(self) -> None:
        assert FinishReason("stop") != FinishReason("length")

    def test_equality_with_string(self) -> None:
        assert FinishReason("stop") == "stop"
        assert FinishReason("tool_calls") == "tool_calls"

    def test_class_constants(self) -> None:
        assert FinishReason.STOP == FinishReason("stop")
        assert FinishReason.TOOL_CALLS == FinishReason("tool_calls")
        assert FinishReason.LENGTH == FinishReason("length")
        assert FinishReason.CONTENT_FILTER == FinishReason("content_filter")
        assert FinishReason.ERROR == FinishReason("error")
        assert FinishReason.OTHER == FinishReason("other")

    def test_tool_use_alias(self) -> None:
        assert FinishReason.TOOL_USE == FinishReason.TOOL_CALLS
        assert FinishReason.TOOL_USE == FinishReason("tool_calls")

    def test_raw_preserved(self) -> None:
        fr = FinishReason("stop", raw="end_turn")
        assert fr.raw == "end_turn"

    def test_str(self) -> None:
        assert str(FinishReason("stop")) == "stop"
        assert str(FinishReason("tool_calls")) == "tool_calls"

    def test_hashable(self) -> None:
        s = {FinishReason("stop"), FinishReason("stop", raw="end_turn")}
        assert len(s) == 1

    def test_repr_without_raw(self) -> None:
        fr = FinishReason("stop")
        assert "stop" in repr(fr)

    def test_repr_with_raw(self) -> None:
        fr = FinishReason("stop", raw="end_turn")
        assert "end_turn" in repr(fr)


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class TestRequestResponse:
    def test_request_defaults(self) -> None:
        req = Request(model="gpt-4o")
        assert req.messages == []
        assert req.tools == []
        assert req.temperature is None

    def test_request_new_fields(self) -> None:
        req = Request(
            model="claude-opus-4-6",
            provider="anthropic",
            tool_choice="auto",
            provider_options={"anthropic": {"thinking": True}},
        )
        assert req.provider == "anthropic"
        assert req.tool_choice == "auto"
        assert req.provider_options == {"anthropic": {"thinking": True}}

    def test_request_provider_options_default_none(self) -> None:
        req = Request(model="test")
        assert req.provider_options is None
        assert req.provider is None
        assert req.tool_choice is None

    def test_response_defaults(self) -> None:
        resp = Response()
        assert resp.message.role == Role.ASSISTANT
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.total_tokens == 0

    def test_response_new_fields(self) -> None:
        from attractor.llm.models import RateLimitInfo

        resp = Response(
            provider="anthropic",
            raw={"id": "msg_123"},
            warnings=["high token usage"],
            rate_limit=RateLimitInfo(limit=100, remaining=50, reset_seconds=30.0),
        )
        assert resp.provider == "anthropic"
        assert resp.raw == {"id": "msg_123"}
        assert resp.warnings == ["high token usage"]
        assert resp.rate_limit is not None
        assert resp.rate_limit.limit == 100
        assert resp.rate_limit.remaining == 50

    def test_response_defaults_empty_collections(self) -> None:
        resp = Response()
        assert resp.provider == ""
        assert resp.raw is None
        assert resp.warnings == []
        assert resp.rate_limit is None

    def test_response_text_property(self) -> None:
        resp = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[TextContent(text="Hello "), TextContent(text="world")],
            )
        )
        assert resp.text == "Hello world"

    def test_response_text_property_empty(self) -> None:
        resp = Response()
        assert resp.text == ""

    def test_response_tool_calls_property(self) -> None:
        tc1 = ToolCallContent(tool_name="search", arguments={"q": "test"})
        tc2 = ToolCallContent(tool_name="read", arguments={"path": "/tmp"})
        resp = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[TextContent(text="let me help"), tc1, tc2],
            )
        )
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].tool_name == "search"
        assert resp.tool_calls[1].tool_name == "read"

    def test_response_tool_calls_property_empty(self) -> None:
        resp = Response(message=Message.assistant("no tools"))
        assert resp.tool_calls == []

    def test_response_reasoning_property(self) -> None:
        resp = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ThinkingContent(text="Step 1. "),
                    ThinkingContent(text="Step 2."),
                    TextContent(text="The answer is 42."),
                ],
            )
        )
        assert resp.reasoning == "Step 1. Step 2."

    def test_response_reasoning_property_none(self) -> None:
        resp = Response(message=Message.assistant("no thinking"))
        assert resp.reasoning is None

    def test_response_reasoning_with_empty_thinking(self) -> None:
        resp = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[ThinkingContent(text=""), TextContent(text="answer")],
            )
        )
        assert resp.reasoning is None


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_text_delta(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_DELTA, text="hello")
        assert evt.type == StreamEventType.TEXT_DELTA
        assert evt.text == "hello"
        assert evt.tool_call is None

    def test_finish_event(self) -> None:
        evt = StreamEvent(
            type=StreamEventType.FINISH,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        assert evt.finish_reason == FinishReason.STOP
        assert evt.usage is not None
        assert evt.usage.total_tokens == 15

    def test_new_stream_event_types_exist(self) -> None:
        """Verify the new stream event types from the spec are available."""
        assert StreamEventType.TEXT_START == "text_start"
        assert StreamEventType.TEXT_END == "text_end"
        assert StreamEventType.THINKING_START == "thinking_start"
        assert StreamEventType.THINKING_END == "thinking_end"

    def test_text_start_event(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_START)
        assert evt.type == StreamEventType.TEXT_START
        assert evt.text == ""

    def test_text_end_event(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_END)
        assert evt.type == StreamEventType.TEXT_END
