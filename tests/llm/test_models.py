"""Tests for attractor.llm.models."""

from __future__ import annotations


import pytest

from attractor.llm.models import (
    ContentKind,
    FinishReason,
    GenerateResult,
    ImageContent,
    Message,
    ModelInfo,
    RateLimitInfo,
    Request,
    Response,
    RetryPolicy,
    Role,
    StepResult,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    TimeoutConfig,
    ToolCallContent,
    ToolChoice,
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
        resp = Response(
            provider="anthropic",
            raw={"id": "msg_123"},
            warnings=["high token usage"],
            rate_limit=RateLimitInfo(
                requests_limit=100, requests_remaining=50, reset_at=1700000000.0
            ),
        )
        assert resp.provider == "anthropic"
        assert resp.raw == {"id": "msg_123"}
        assert resp.warnings == ["high token usage"]
        assert resp.rate_limit is not None
        assert resp.rate_limit.requests_limit == 100
        assert resp.rate_limit.requests_remaining == 50

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


# ---------------------------------------------------------------------------
# TokenUsage.__add__
# ---------------------------------------------------------------------------


class TestTokenUsageAdd:
    def test_sum_all_fields(self) -> None:
        a = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )
        b = TokenUsage(
            input_tokens=200,
            output_tokens=80,
            reasoning_tokens=30,
            cache_read_tokens=15,
            cache_write_tokens=10,
        )
        result = a + b
        assert result.input_tokens == 300
        assert result.output_tokens == 130
        assert result.reasoning_tokens == 50
        assert result.cache_read_tokens == 25
        assert result.cache_write_tokens == 15

    def test_zero_plus_nonzero(self) -> None:
        zero = TokenUsage()
        nonzero = TokenUsage(input_tokens=42, output_tokens=17)
        result = zero + nonzero
        assert result.input_tokens == 42
        assert result.output_tokens == 17

    def test_commutativity(self) -> None:
        a = TokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
        b = TokenUsage(input_tokens=30, output_tokens=40, reasoning_tokens=15)
        assert (a + b).input_tokens == (b + a).input_tokens
        assert (a + b).output_tokens == (b + a).output_tokens
        assert (a + b).reasoning_tokens == (b + a).reasoning_tokens

    def test_result_is_new_instance(self) -> None:
        a = TokenUsage(input_tokens=10)
        b = TokenUsage(input_tokens=20)
        result = a + b
        assert result is not a
        assert result is not b
        assert result.input_tokens == 30

    def test_total_tokens_after_add(self) -> None:
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=80)
        result = a + b
        assert result.total_tokens == 430


# ---------------------------------------------------------------------------
# ToolChoice
# ---------------------------------------------------------------------------


class TestToolChoice:
    def test_auto_mode(self) -> None:
        tc = ToolChoice(mode="auto")
        assert tc.mode == "auto"
        assert tc.tool_name is None

    def test_none_mode(self) -> None:
        tc = ToolChoice(mode="none")
        assert tc.mode == "none"

    def test_required_mode(self) -> None:
        tc = ToolChoice(mode="required")
        assert tc.mode == "required"

    def test_named_mode_with_tool(self) -> None:
        tc = ToolChoice(mode="named", tool_name="search")
        assert tc.mode == "named"
        assert tc.tool_name == "search"

    def test_named_mode_without_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_name is required"):
            ToolChoice(mode="named")

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid tool choice mode"):
            ToolChoice(mode="always")

    def test_default_mode_is_auto(self) -> None:
        tc = ToolChoice()
        assert tc.mode == "auto"


# ---------------------------------------------------------------------------
# GenerateResult and StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_defaults(self) -> None:
        step = StepResult()
        assert step.text == ""
        assert step.reasoning is None
        assert step.tool_calls == []
        assert step.tool_results == []
        assert step.finish_reason == "stop"
        assert step.usage.total_tokens == 0
        assert step.response is None
        assert step.warnings == []

    def test_construction_with_fields(self) -> None:
        tc = ToolCallContent(tool_name="read", arguments={"path": "/tmp"})
        tr = ToolResultContent(tool_call_id="call_1", content="data")
        step = StepResult(
            text="result",
            reasoning="I thought about it",
            tool_calls=[tc],
            tool_results=[tr],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            warnings=["token limit close"],
        )
        assert step.text == "result"
        assert step.reasoning == "I thought about it"
        assert len(step.tool_calls) == 1
        assert len(step.tool_results) == 1
        assert step.usage.total_tokens == 15
        assert step.warnings == ["token limit close"]


class TestGenerateResult:
    def test_defaults(self) -> None:
        gr = GenerateResult()
        assert gr.text == ""
        assert gr.reasoning is None
        assert gr.tool_calls == []
        assert gr.tool_results == []
        assert gr.finish_reason == "stop"
        assert gr.usage.total_tokens == 0
        assert gr.total_usage.total_tokens == 0
        assert gr.steps == []
        assert gr.response is None
        assert gr.output is None

    def test_steps_list(self) -> None:
        s1 = StepResult(text="step 1", usage=TokenUsage(input_tokens=10))
        s2 = StepResult(text="step 2", usage=TokenUsage(input_tokens=20))
        gr = GenerateResult(
            text="final",
            steps=[s1, s2],
            total_usage=s1.usage + s2.usage,
        )
        assert len(gr.steps) == 2
        assert gr.total_usage.input_tokens == 30

    def test_output_field(self) -> None:
        gr = GenerateResult(output={"key": "value"})
        assert gr.output == {"key": "value"}

    def test_output_arbitrary_types(self) -> None:
        gr = GenerateResult(output=[1, 2, 3])
        assert gr.output == [1, 2, 3]


# ---------------------------------------------------------------------------
# StreamEvent new fields
# ---------------------------------------------------------------------------


class TestStreamEventFields:
    def test_delta_field(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hello")
        assert evt.delta == "hello"

    def test_text_id_field(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_START, text_id="txt_0")
        assert evt.text_id == "txt_0"

    def test_reasoning_delta_field(self) -> None:
        evt = StreamEvent(
            type=StreamEventType.REASONING_DELTA, reasoning_delta="thinking..."
        )
        assert evt.reasoning_delta == "thinking..."

    def test_raw_field(self) -> None:
        evt = StreamEvent(
            type=StreamEventType.PROVIDER_EVENT,
            raw={"provider": "anthropic", "event": "ping"},
        )
        assert evt.raw == {"provider": "anthropic", "event": "ping"}

    def test_new_fields_default_none(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_DELTA, text="hi")
        assert evt.delta is None
        assert evt.text_id is None
        assert evt.reasoning_delta is None
        assert evt.raw is None


# ---------------------------------------------------------------------------
# StreamEventType new values
# ---------------------------------------------------------------------------


class TestStreamEventTypeNewValues:
    def test_reasoning_start_exists(self) -> None:
        assert StreamEventType.REASONING_START == "thinking_start"

    def test_reasoning_end_exists(self) -> None:
        assert StreamEventType.REASONING_END == "thinking_end"

    def test_provider_event_exists(self) -> None:
        assert StreamEventType.PROVIDER_EVENT == "provider_event"

    def test_step_finish_exists(self) -> None:
        assert StreamEventType.STEP_FINISH == "step_finish"

    def test_reasoning_start_is_thinking_start(self) -> None:
        assert StreamEventType.REASONING_START is StreamEventType.THINKING_START

    def test_reasoning_end_is_thinking_end(self) -> None:
        assert StreamEventType.REASONING_END is StreamEventType.THINKING_END


# ---------------------------------------------------------------------------
# RateLimitInfo restructured fields
# ---------------------------------------------------------------------------


class TestRateLimitInfo:
    def test_defaults_all_none(self) -> None:
        rl = RateLimitInfo()
        assert rl.requests_remaining is None
        assert rl.requests_limit is None
        assert rl.tokens_remaining is None
        assert rl.tokens_limit is None
        assert rl.reset_at is None

    def test_construction_with_values(self) -> None:
        rl = RateLimitInfo(
            requests_remaining=99,
            requests_limit=100,
            tokens_remaining=50000,
            tokens_limit=100000,
            reset_at=1700000000.0,
        )
        assert rl.requests_remaining == 99
        assert rl.requests_limit == 100
        assert rl.tokens_remaining == 50000
        assert rl.tokens_limit == 100000
        assert rl.reset_at == 1700000000.0

    def test_partial_construction(self) -> None:
        rl = RateLimitInfo(requests_remaining=50)
        assert rl.requests_remaining == 50
        assert rl.tokens_remaining is None


# ---------------------------------------------------------------------------
# ImageContent.detail
# ---------------------------------------------------------------------------


class TestImageContentDetail:
    def test_detail_default_none(self) -> None:
        img = ImageContent(url="https://example.com/img.png")
        assert img.detail is None

    def test_detail_set_to_auto(self) -> None:
        img = ImageContent(url="https://example.com/img.png", detail="auto")
        assert img.detail == "auto"

    def test_detail_set_to_high(self) -> None:
        img = ImageContent(url="https://example.com/img.png", detail="high")
        assert img.detail == "high"

    def test_detail_set_to_low(self) -> None:
        img = ImageContent(url="https://example.com/img.png", detail="low")
        assert img.detail == "low"


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    def test_defaults_none(self) -> None:
        tc = TimeoutConfig()
        assert tc.total is None
        assert tc.per_step is None

    def test_total_only(self) -> None:
        tc = TimeoutConfig(total=120.0)
        assert tc.total == 120.0
        assert tc.per_step is None

    def test_per_step_only(self) -> None:
        tc = TimeoutConfig(per_step=30.0)
        assert tc.total is None
        assert tc.per_step == 30.0

    def test_both_set(self) -> None:
        tc = TimeoutConfig(total=300.0, per_step=60.0)
        assert tc.total == 300.0
        assert tc.per_step == 60.0


# ---------------------------------------------------------------------------
# ModelInfo.aliases
# ---------------------------------------------------------------------------


class TestModelInfoAliases:
    def test_aliases_default_empty(self) -> None:
        mi = ModelInfo(model_id="claude-opus-4-6", provider="anthropic")
        assert mi.aliases == []

    def test_aliases_set(self) -> None:
        mi = ModelInfo(
            model_id="claude-sonnet-4-20250514",
            provider="anthropic",
            aliases=["sonnet", "claude-sonnet"],
        )
        assert mi.aliases == ["sonnet", "claude-sonnet"]
        assert len(mi.aliases) == 2

    def test_aliases_independent_instances(self) -> None:
        m1 = ModelInfo(model_id="a", provider="p")
        m2 = ModelInfo(model_id="b", provider="p")
        m1.aliases.append("test")
        assert m2.aliases == []
