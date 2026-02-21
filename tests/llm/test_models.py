"""Tests for attractor.llm.models."""

from __future__ import annotations

import pytest

from attractor.llm.models import (
    ContentKind,
    FinishReason,
    ImageContent,
    Message,
    ReasoningEffort,
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
# Request / Response
# ---------------------------------------------------------------------------


class TestRequestResponse:
    def test_request_defaults(self) -> None:
        req = Request(model="gpt-4o")
        assert req.messages == []
        assert req.tools == []
        assert req.temperature is None

    def test_response_defaults(self) -> None:
        resp = Response()
        assert resp.message.role == Role.ASSISTANT
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.total_tokens == 0


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
