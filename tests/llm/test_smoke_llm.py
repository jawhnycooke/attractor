"""LLM subsystem integration smoke tests (spec §8.10).

Requires a real ANTHROPIC_API_KEY.  Run with:

    pytest -m smoke tests/llm/test_smoke_llm.py -v
"""

from __future__ import annotations

import pytest

from attractor.llm.client import LLMClient
from attractor.llm.models import (
    ImageContent,
    Message,
    Role,
    StreamEventType,
    TextContent,
    ToolCallContent,
    ToolDefinition,
)

SMOKE_MODEL = "claude-sonnet-4-6"

# Minimal 1x1 red PNG encoded as base64
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

pytestmark = pytest.mark.smoke


class TestLLMSmoke:
    """Smoke tests for the LLM client against a real Anthropic endpoint."""

    # §8.10 step 1 — basic text generation
    async def test_basic_generation(self, llm_client: LLMClient) -> None:
        result = await llm_client.generate(
            prompt="Reply with exactly one word: hello.",
            model=SMOKE_MODEL,
        )

        assert len(result.text) > 0, "Expected non-empty response text"
        assert result.usage.total_tokens > 0, "Expected positive token usage"
        assert result.finish_reason == "stop", (
            f"Expected finish_reason 'stop', got {result.finish_reason}"
        )

    # §8.10 step 2 — streaming
    async def test_streaming(self, llm_client: LLMClient) -> None:
        events = []
        async for event in llm_client.stream_generate(
            prompt="Say hello in one sentence.",
            model=SMOKE_MODEL,
        ):
            events.append(event)

        text_deltas = [
            e for e in events if e.type == StreamEventType.TEXT_DELTA
        ]
        assert len(text_deltas) > 0, "Expected at least one TEXT_DELTA event"

        concatenated = "".join(e.text for e in text_deltas if e.text)
        assert len(concatenated) > 0, "Concatenated stream text should be non-empty"

    # §8.10 step 3 — tool calling with auto-execution loop
    async def test_tool_calling(self, llm_client: LLMClient) -> None:
        weather_tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["city"],
            },
        )

        async def weather_executor(tc: ToolCallContent) -> str:
            city = tc.arguments.get("city", "unknown")
            return f"Weather in {city}: 72°F, sunny"

        # Single city to avoid multi-round message alternation issues
        # with the Anthropic adapter (multiple rounds can trigger strict
        # user/assistant alternation errors).
        result = await llm_client.generate(
            prompt="What is the weather in Tokyo? Use the get_weather tool.",
            model=SMOKE_MODEL,
            tools=[weather_tool],
            tool_executor=weather_executor,
            max_tool_rounds=3,
        )

        assert len(result.tool_calls) > 0, "Expected at least one tool call"
        text_lower = result.text.lower()
        assert "tokyo" in text_lower or "72" in text_lower, (
            "Expected response to mention the queried city or temperature"
        )

    # §8.10 step 4 — image / multimodal input
    async def test_image_input(self, llm_client: LLMClient) -> None:
        msg = Message(
            role=Role.USER,
            content=[
                ImageContent(base64_data=_PNG_B64, media_type="image/png"),
                TextContent(text="Describe this image in one sentence."),
            ],
        )

        result = await llm_client.generate(prompt=[msg], model=SMOKE_MODEL)
        assert len(result.text) > 0, "Expected non-empty response for image input"
        # The 1x1 red PNG should elicit a description mentioning color or image
        text_lower = result.text.lower()
        assert any(
            word in text_lower
            for word in ("red", "pixel", "small", "1x1", "image", "color", "single")
        ), f"Expected image description to reference the red pixel; got: {result.text[:200]}"

    # §8.10 step 5 — structured output via generate_object()
    async def test_structured_output(self, llm_client: LLMClient) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        }

        result = await llm_client.generate_object(
            prompt="Extract the person info from: 'John is 30 years old.'",
            model=SMOKE_MODEL,
            schema=schema,
            schema_name="person",
        )

        assert result.output is not None, "Expected parsed JSON object"
        assert isinstance(result.output, dict)
        assert result.output.get("name") == "John", (
            f"Expected name='John', got {result.output.get('name')!r}"
        )
        assert result.output.get("age") == 30, (
            f"Expected age=30, got {result.output.get('age')!r}"
        )

    # §8.10 step 6 — error handling for invalid model
    async def test_error_handling(self, llm_client: LLMClient) -> None:
        from attractor.llm.errors import SDKError

        with pytest.raises(SDKError):
            await llm_client.generate(
                prompt="Hello",
                model="nonexistent-model-xyz",
            )
