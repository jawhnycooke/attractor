"""Multi-provider LLM smoke tests.

Tests basic generation, streaming, tool calling, and structured output
across Anthropic, OpenAI, and Gemini providers.

Requires real API keys in .env.local.  Run with:

    pytest -m smoke tests/llm/test_smoke_providers.py -v
"""

from __future__ import annotations

import pytest

from attractor.llm.client import LLMClient
from attractor.llm.errors import ProviderError
from attractor.llm.models import (
    StreamEventType,
    ToolCallContent,
    ToolDefinition,
)

pytestmark = pytest.mark.smoke


# ── Provider configurations ────────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "fixture": "requires_anthropic_key",
        "supports_vision": True,
    },
    "openai": {
        "model": "gpt-4.1",
        "fixture": "requires_openai_key",
        "supports_vision": True,
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "fixture": "requires_gemini_key",
        "supports_vision": True,
    },
}


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> LLMClient:
    """Return an LLMClient with all available adapters."""
    return LLMClient.from_env()


# ── Anthropic smoke tests ─────────────────────────────────────────────


class TestAnthropicSmoke:
    """Smoke tests against the real Anthropic API."""

    MODEL = PROVIDERS["anthropic"]["model"]

    async def test_basic_generation(
        self, requires_anthropic_key: None, client: LLMClient
    ) -> None:
        result = await client.generate(
            prompt="Reply with exactly one word: hello.",
            model=self.MODEL,
        )
        assert len(result.text) > 0
        assert result.usage.total_tokens > 0
        assert result.finish_reason == "stop", (
            f"Expected finish_reason 'stop', got {result.finish_reason}"
        )

    async def test_streaming(
        self, requires_anthropic_key: None, client: LLMClient
    ) -> None:
        events = []
        async for event in client.stream_generate(
            prompt="Say hello in one sentence.",
            model=self.MODEL,
        ):
            events.append(event)

        text_deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert len(text_deltas) > 0
        concatenated = "".join(e.text for e in text_deltas if e.text)
        assert len(concatenated) > 0

    async def test_tool_calling(
        self, requires_anthropic_key: None, client: LLMClient
    ) -> None:
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )

        async def executor(tc: ToolCallContent) -> str:
            city = tc.arguments.get("city", "unknown")
            return f"Weather in {city}: 72°F, sunny"

        result = await client.generate(
            prompt="What is the weather in Tokyo? Use the get_weather tool.",
            model=self.MODEL,
            tools=[tool],
            tool_executor=executor,
            max_tool_rounds=3,
        )
        assert len(result.tool_calls) > 0
        text_lower = result.text.lower()
        assert "tokyo" in text_lower or "72" in text_lower

    async def test_structured_output(
        self, requires_anthropic_key: None, client: LLMClient
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await client.generate_object(
            prompt="Extract: 'Alice is 25 years old.'",
            model=self.MODEL,
            schema=schema,
            schema_name="person",
        )
        assert isinstance(result.output, dict)
        assert result.output.get("name") == "Alice", (
            f"Expected name='Alice', got {result.output.get('name')!r}"
        )
        assert result.output.get("age") == 25, (
            f"Expected age=25, got {result.output.get('age')!r}"
        )


# ── OpenAI smoke tests ────────────────────────────────────────────────


class TestOpenAISmoke:
    """Smoke tests against the real OpenAI API."""

    MODEL = PROVIDERS["openai"]["model"]

    async def test_basic_generation(
        self, requires_openai_key: None, client: LLMClient
    ) -> None:
        result = await client.generate(
            prompt="Reply with exactly one word: hello.",
            model=self.MODEL,
        )
        assert len(result.text) > 0
        assert result.usage.total_tokens > 0
        assert result.finish_reason == "stop", (
            f"Expected finish_reason 'stop', got {result.finish_reason}"
        )

    async def test_streaming(
        self, requires_openai_key: None, client: LLMClient
    ) -> None:
        events = []
        async for event in client.stream_generate(
            prompt="Say hello in one sentence.",
            model=self.MODEL,
        ):
            events.append(event)

        text_deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert len(text_deltas) > 0
        concatenated = "".join(e.text for e in text_deltas if e.text)
        assert len(concatenated) > 0

    async def test_tool_calling(
        self, requires_openai_key: None, client: LLMClient
    ) -> None:
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )

        async def executor(tc: ToolCallContent) -> str:
            city = tc.arguments.get("city", "unknown")
            return f"Weather in {city}: 72°F, sunny"

        result = await client.generate(
            prompt="What is the weather in Tokyo? Use the get_weather tool.",
            model=self.MODEL,
            tools=[tool],
            tool_executor=executor,
            max_tool_rounds=3,
        )
        assert len(result.tool_calls) > 0
        text_lower = result.text.lower()
        assert "tokyo" in text_lower or "72" in text_lower

    async def test_structured_output(
        self, requires_openai_key: None, client: LLMClient
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await client.generate_object(
            prompt="Extract: 'Bob is 30 years old.'",
            model=self.MODEL,
            schema=schema,
            schema_name="person",
        )
        assert isinstance(result.output, dict)
        assert result.output.get("name") == "Bob", (
            f"Expected name='Bob', got {result.output.get('name')!r}"
        )
        assert result.output.get("age") == 30, (
            f"Expected age=30, got {result.output.get('age')!r}"
        )


# ── Gemini smoke tests ────────────────────────────────────────────────


class TestGeminiSmoke:
    """Smoke tests against the real Google Gemini API."""

    MODEL = PROVIDERS["gemini"]["model"]

    async def test_basic_generation(
        self, requires_gemini_key: None, client: LLMClient
    ) -> None:
        result = await client.generate(
            prompt="Reply with exactly one word: hello.",
            model=self.MODEL,
        )
        assert len(result.text) > 0
        assert result.usage.total_tokens > 0
        assert result.finish_reason == "stop", (
            f"Expected finish_reason 'stop', got {result.finish_reason}"
        )

    async def test_streaming(
        self, requires_gemini_key: None, client: LLMClient
    ) -> None:
        events = []
        async for event in client.stream_generate(
            prompt="Say hello in one sentence.",
            model=self.MODEL,
        ):
            events.append(event)

        text_deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert len(text_deltas) > 0
        concatenated = "".join(e.text for e in text_deltas if e.text)
        assert len(concatenated) > 0

    async def test_tool_calling(
        self, requires_gemini_key: None, client: LLMClient
    ) -> None:
        tool = ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )

        async def executor(tc: ToolCallContent) -> str:
            city = tc.arguments.get("city", "unknown")
            return f"Weather in {city}: 72°F, sunny"

        result = await client.generate(
            prompt="What is the weather in Tokyo? Use the get_weather tool.",
            model=self.MODEL,
            tools=[tool],
            tool_executor=executor,
            max_tool_rounds=3,
        )
        assert len(result.tool_calls) > 0
        text_lower = result.text.lower()
        assert "tokyo" in text_lower or "72" in text_lower

    async def test_structured_output(
        self, requires_gemini_key: None, client: LLMClient
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = await client.generate_object(
            prompt="Extract: 'Carol is 28 years old.'",
            model=self.MODEL,
            schema=schema,
            schema_name="person",
        )
        assert isinstance(result.output, dict)
        assert result.output.get("name") == "Carol", (
            f"Expected name='Carol', got {result.output.get('name')!r}"
        )
        assert result.output.get("age") == 28, (
            f"Expected age=28, got {result.output.get('age')!r}"
        )


# ── Cross-provider: error handling ─────────────────────────────────────


class TestErrorHandling:
    """Error handling should work consistently across providers."""

    async def test_invalid_model_anthropic(
        self, requires_anthropic_key: None, client: LLMClient
    ) -> None:
        with pytest.raises(ProviderError):
            await client.generate(prompt="Hello", model="claude-nonexistent-xyz")

    async def test_invalid_model_openai(
        self, requires_openai_key: None, client: LLMClient
    ) -> None:
        with pytest.raises(ProviderError):
            await client.generate(prompt="Hello", model="gpt-nonexistent-xyz")

    async def test_invalid_model_gemini(
        self, requires_gemini_key: None, client: LLMClient
    ) -> None:
        with pytest.raises(ProviderError):
            await client.generate(prompt="Hello", model="gemini-nonexistent-xyz")
