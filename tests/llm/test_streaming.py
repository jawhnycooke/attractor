"""Tests for StreamCollector — streaming event aggregation."""

import logging
from collections.abc import AsyncIterator

import pytest

from attractor.llm.models import (
    FinishReason,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCallContent,
)
from attractor.llm.streaming import StreamCollector


class TestTextAccumulation:
    def test_text_delta_accumulation_preserves_whitespace(self) -> None:
        collector = StreamCollector()
        collector.process_event(
            StreamEvent(type=StreamEventType.TEXT_DELTA, text="  hello  ")
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TEXT_DELTA, text="\n  world  \n")
        )

        response = collector.to_response()
        text = response.message.text()
        assert text == "  hello  \n  world  \n"


class TestToolCallLifecycle:
    def test_tool_call_full_lifecycle(self) -> None:
        collector = StreamCollector()

        tc = ToolCallContent(
            tool_call_id="call_123",
            tool_name="read_file",
        )

        # START
        collector.process_event(
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call=tc,
            )
        )

        # DELTA — stream the JSON arguments
        collector.process_event(
            StreamEvent(
                type=StreamEventType.TOOL_CALL_DELTA,
                text='{"path": "foo',
            )
        )
        collector.process_event(
            StreamEvent(
                type=StreamEventType.TOOL_CALL_DELTA,
                text='.py"}',
            )
        )

        # END — finalize with the full tool call
        final_tc = ToolCallContent(
            tool_call_id="call_123",
            tool_name="read_file",
            arguments_json='{"path": "foo.py"}',
        )
        collector.process_event(
            StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                tool_call=final_tc,
            )
        )

        response = collector.to_response()
        calls = response.message.tool_calls()
        assert len(calls) == 1
        assert calls[0].tool_call_id == "call_123"
        assert calls[0].tool_name == "read_file"
        assert calls[0].arguments == {"path": "foo.py"}

    def test_multiple_tool_calls_in_sequence(self) -> None:
        collector = StreamCollector()

        # First tool call
        tc1 = ToolCallContent(tool_call_id="call_1", tool_name="read_file")
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc1)
        )
        final_tc1 = ToolCallContent(
            tool_call_id="call_1",
            tool_name="read_file",
            arguments_json='{"path": "a.py"}',
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=final_tc1)
        )

        # Second tool call
        tc2 = ToolCallContent(tool_call_id="call_2", tool_name="write_file")
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc2)
        )
        final_tc2 = ToolCallContent(
            tool_call_id="call_2",
            tool_name="write_file",
            arguments_json='{"path": "b.py", "content": "hello"}',
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=final_tc2)
        )

        response = collector.to_response()
        calls = response.message.tool_calls()
        assert len(calls) == 2
        assert calls[0].tool_name == "read_file"
        assert calls[1].tool_name == "write_file"


class TestMalformedJson:
    def test_malformed_json_logs_warning_and_uses_empty_dict(self, caplog) -> None:
        """Regression for C3: bad JSON in tool call arguments should warn
        and not crash."""
        collector = StreamCollector()

        tc = ToolCallContent(
            tool_call_id="call_bad",
            tool_name="shell",
        )
        collector.process_event(
            StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call=tc)
        )

        # End with malformed JSON
        bad_tc = ToolCallContent(
            tool_call_id="call_bad",
            tool_name="shell",
            arguments_json="{not valid json!!!}",
            arguments={},  # empty — post_init won't parse bad json
        )
        # Override arguments_json after init so __post_init__ doesn't fix it
        bad_tc.arguments_json = "{not valid json!!!}"
        bad_tc.arguments = {}

        with caplog.at_level(logging.WARNING, logger="attractor.llm.streaming"):
            collector.process_event(
                StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=bad_tc)
            )

        # The warning should have been logged
        assert any("parse tool call arguments" in r.message for r in caplog.records)

        # The tool call should still be in the result
        response = collector.to_response()
        calls = response.message.tool_calls()
        assert len(calls) == 1


class TestUsageMerge:
    def test_usage_merge_from_finish_event(self) -> None:
        collector = StreamCollector()

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=10,
        )
        collector.process_event(
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason.STOP,
                usage=usage,
            )
        )

        response = collector.to_response()
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.reasoning_tokens == 10
        assert response.finish_reason == FinishReason.STOP


class TestCollectAsyncIterator:
    @pytest.mark.asyncio
    async def test_collect_consumes_async_iterator(self) -> None:
        async def event_stream() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type=StreamEventType.STREAM_START)
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="Hello ")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="World")
            yield StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )

        collector = StreamCollector()
        response = await collector.collect(event_stream())

        assert response.message.text() == "Hello World"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
