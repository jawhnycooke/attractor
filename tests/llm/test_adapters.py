"""Tests for provider adapter enhancements.

Tests provider_options support (spec §2.8) and reasoning token
extraction (spec §3.9) across Anthropic, OpenAI, and Gemini adapters.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from attractor.llm.models import (
    Message,
    ReasoningEffort,
    Request,
    Role,
    ToolCallContent,
    ToolResultContent,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


async def _async_return(value):
    """Wrap a value in an awaitable for mocking async SDK calls."""
    return value


def _simple_request(**overrides: object) -> Request:
    """Build a minimal Request with optional overrides."""
    defaults = {
        "model": "test-model",
        "messages": [Message.user("hello")],
    }
    defaults.update(overrides)
    return Request(**defaults)  # type: ignore[arg-type]


# ===============================================================
# Anthropic adapter
# ===============================================================


def _make_anthropic_adapter():
    """Instantiate AnthropicAdapter without a real SDK client."""
    try:
        from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter
    except ImportError:
        pytest.skip("anthropic not installed")

    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter._client = MagicMock()
    adapter._api_key = "test-key"
    return adapter


class TestAnthropicProviderOptions:
    """Verify _build_kwargs reads anthropic-specific provider_options."""

    def test_beta_headers_injected(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(
            model="claude-sonnet-4-20250514",
            provider_options={
                "anthropic": {
                    "beta_headers": ["pdfs-2024-09-25", "output-128k-2025-02-19"],
                }
            },
        )
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" in kwargs
        header_val = kwargs["extra_headers"]["anthropic-beta"]
        assert "pdfs-2024-09-25" in header_val
        assert "output-128k-2025-02-19" in header_val

    def test_thinking_override_replaces_reasoning_effort(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(
            model="claude-sonnet-4-20250514",
            reasoning_effort=ReasoningEffort.LOW,
            provider_options={
                "anthropic": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 99999,
                    },
                }
            },
        )
        kwargs = adapter._build_kwargs(request)
        # The provider_options override should replace the reasoning_effort value
        assert kwargs["thinking"]["budget_tokens"] == 99999

    def test_no_provider_options_leaves_kwargs_unchanged(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(model="claude-sonnet-4-20250514")
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs

    def test_empty_anthropic_opts_is_harmless(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(
            model="claude-sonnet-4-20250514",
            provider_options={"anthropic": {}},
        )
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs

    def test_non_dict_anthropic_opts_ignored(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(
            model="claude-sonnet-4-20250514",
            provider_options={"anthropic": "not-a-dict"},
        )
        # Should not raise
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs

    def test_other_provider_key_ignored(self) -> None:
        adapter = _make_anthropic_adapter()
        request = _simple_request(
            model="claude-sonnet-4-20250514",
            provider_options={"openai": {"extra_body": {"foo": 1}}},
        )
        kwargs = adapter._build_kwargs(request)
        assert "foo" not in kwargs


class TestAnthropicReasoningTokens:
    """Verify _map_response estimates reasoning tokens from thinking blocks."""

    def _mock_response(self, thinking_text: str | None = None):
        """Build a mock Anthropic raw response object."""
        blocks = []
        if thinking_text:
            tb = MagicMock()
            tb.type = "thinking"
            tb.thinking = thinking_text
            tb.signature = None
            blocks.append(tb)

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "response text"
        blocks.append(text_block)

        raw = MagicMock()
        raw.content = blocks
        raw.usage.input_tokens = 100
        raw.usage.output_tokens = 50
        raw.usage.cache_read_input_tokens = 0
        raw.usage.cache_creation_input_tokens = 0
        raw.stop_reason = "end_turn"
        raw.model = "claude-sonnet-4-20250514"
        raw.id = "msg_test123"
        return raw

    def test_reasoning_tokens_estimated_from_thinking(self) -> None:
        adapter = _make_anthropic_adapter()
        thinking = "Let me think about this problem step by step and consider all angles"
        raw = self._mock_response(thinking_text=thinking)
        response = adapter._map_response(raw, latency_ms=100.0)

        # "Let me think about this problem step by step and consider all angles"
        # = 13 words × 1.3 = 16.9 → int(16.9) = 16
        assert response.usage.reasoning_tokens == 16

    def test_no_thinking_yields_zero_reasoning_tokens(self) -> None:
        adapter = _make_anthropic_adapter()
        raw = self._mock_response(thinking_text=None)
        response = adapter._map_response(raw, latency_ms=100.0)
        assert response.usage.reasoning_tokens == 0

    def test_cache_tokens_still_populated(self) -> None:
        adapter = _make_anthropic_adapter()
        raw = self._mock_response()
        raw.usage.cache_read_input_tokens = 42
        raw.usage.cache_creation_input_tokens = 7
        response = adapter._map_response(raw, latency_ms=50.0)
        assert response.usage.cache_read_tokens == 42
        assert response.usage.cache_write_tokens == 7


# ===============================================================
# OpenAI adapter
# ===============================================================


def _make_openai_adapter():
    """Instantiate OpenAIAdapter without a real SDK client."""
    try:
        from attractor.llm.adapters.openai_adapter import OpenAIAdapter
    except ImportError:
        pytest.skip("openai not installed")

    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter._client = MagicMock()
    return adapter


class TestOpenAIProviderOptions:
    """Verify _build_kwargs reads openai-specific provider_options."""

    def test_extra_body_merged_into_kwargs(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(
            model="gpt-4o",
            provider_options={
                "openai": {
                    "extra_body": {"logprobs": True, "top_logprobs": 5},
                }
            },
        )
        kwargs = adapter._build_kwargs(request)
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 5

    def test_extra_headers_set(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(
            model="gpt-4o",
            provider_options={
                "openai": {
                    "extra_headers": {"X-Custom": "value"},
                }
            },
        )
        kwargs = adapter._build_kwargs(request)
        assert kwargs["extra_headers"] == {"X-Custom": "value"}

    def test_no_provider_options_no_extra_keys(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(model="gpt-4o")
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs
        assert "logprobs" not in kwargs

    def test_empty_openai_opts_harmless(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(
            model="gpt-4o",
            provider_options={"openai": {}},
        )
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs

    def test_non_dict_openai_opts_ignored(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(
            model="gpt-4o",
            provider_options={"openai": 123},
        )
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs

    def test_other_provider_key_ignored(self) -> None:
        adapter = _make_openai_adapter()
        request = _simple_request(
            model="gpt-4o",
            provider_options={"anthropic": {"beta_headers": ["x"]}},
        )
        kwargs = adapter._build_kwargs(request)
        assert "extra_headers" not in kwargs


class TestOpenAIProviderField:
    """L-F03: Verify Response.provider is populated by the OpenAI adapter."""

    def _mock_response(self):
        text_content = MagicMock()
        text_content.type = "output_text"
        text_content.text = "hello"

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_content]

        raw = MagicMock()
        raw.output = [message_item]
        raw.model = "gpt-4o"
        raw.id = "resp_123"
        raw.status = "completed"
        raw.usage.input_tokens = 10
        raw.usage.output_tokens = 5
        raw.usage.output_tokens_details = None
        raw.usage.input_tokens_details = None
        return raw

    def test_provider_is_openai(self) -> None:
        adapter = _make_openai_adapter()
        raw = self._mock_response()
        response = adapter._map_response(raw, latency_ms=50.0)
        assert response.provider == "openai"


class TestOpenAIReasoningTokens:
    """Verify existing reasoning token extraction in _map_response."""

    def _mock_response(self, reasoning_tokens: int = 0):
        """Build a mock OpenAI Responses API raw response."""
        text_content = MagicMock()
        text_content.type = "output_text"
        text_content.text = "response text"

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_content]

        raw = MagicMock()
        raw.output = [message_item]
        raw.model = "gpt-4o"
        raw.id = "resp_test123"
        raw.status = "completed"

        raw.usage.input_tokens = 100
        raw.usage.output_tokens = 50
        if reasoning_tokens > 0:
            raw.usage.output_tokens_details.reasoning_tokens = reasoning_tokens
        else:
            raw.usage.output_tokens_details = None
        raw.usage.input_tokens_details = None
        return raw

    def test_reasoning_tokens_extracted(self) -> None:
        adapter = _make_openai_adapter()
        raw = self._mock_response(reasoning_tokens=200)
        response = adapter._map_response(raw, latency_ms=80.0)
        assert response.usage.reasoning_tokens == 200

    def test_no_reasoning_details_yields_zero(self) -> None:
        adapter = _make_openai_adapter()
        raw = self._mock_response(reasoning_tokens=0)
        response = adapter._map_response(raw, latency_ms=80.0)
        assert response.usage.reasoning_tokens == 0


# ===============================================================
# Gemini adapter
# ===============================================================


def _make_gemini_adapter():
    """Instantiate GeminiAdapter without a real SDK client."""
    try:
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter
    except ImportError:
        pytest.skip("google-genai not installed")

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter._client = MagicMock()
    adapter._api_key = "test-key"
    return adapter


class TestGeminiProviderOptions:
    """Verify _build_kwargs reads gemini-specific provider_options."""

    def test_safety_settings_injected(self) -> None:
        adapter = _make_gemini_adapter()
        safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
        request = _simple_request(
            model="gemini-2.0-flash",
            provider_options={"gemini": {"safety_settings": safety}},
        )
        kwargs = adapter._build_kwargs(request)
        assert kwargs["config"]["safety_settings"] == safety

    def test_thinking_config_injected(self) -> None:
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            provider_options={
                "gemini": {
                    "thinking_config": {"thinking_budget": 8192},
                }
            },
        )
        kwargs = adapter._build_kwargs(request)
        assert kwargs["config"]["thinking_config"] == {"thinking_budget": 8192}

    def test_cached_content_set_at_top_level(self) -> None:
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            provider_options={
                "gemini": {"cached_content": "cached-content-id-123"},
            },
        )
        kwargs = adapter._build_kwargs(request)
        assert kwargs["cached_content"] == "cached-content-id-123"
        # cached_content is on kwargs, not inside config
        assert "cached_content" not in kwargs.get("config", {})

    def test_no_provider_options_no_extra_keys(self) -> None:
        adapter = _make_gemini_adapter()
        request = _simple_request(model="gemini-2.0-flash")
        kwargs = adapter._build_kwargs(request)
        assert "cached_content" not in kwargs
        config = kwargs.get("config", {})
        assert "safety_settings" not in config
        assert "thinking_config" not in config

    def test_empty_gemini_opts_harmless(self) -> None:
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            provider_options={"gemini": {}},
        )
        kwargs = adapter._build_kwargs(request)
        assert "cached_content" not in kwargs

    def test_non_dict_gemini_opts_ignored(self) -> None:
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            provider_options={"gemini": [1, 2, 3]},
        )
        kwargs = adapter._build_kwargs(request)
        assert "cached_content" not in kwargs


class TestGeminiProviderField:
    """L-F03: Verify Response.provider is populated by the Gemini adapter."""

    def _mock_response(self):
        text_part = MagicMock()
        text_part.function_call = None
        text_part.text = "hello"

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.model_version = "gemini-2.0-flash"
        raw.usage_metadata = None
        return raw

    def test_provider_is_gemini(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response()
        response = adapter._map_response(raw, latency_ms=50.0)
        assert response.provider == "gemini"


class TestGeminiReasoningTokens:
    """Verify _map_response extracts reasoning tokens from usage_metadata."""

    def _mock_response(self, thoughts_token_count: int | None = None):
        """Build a mock Gemini raw response."""
        text_part = MagicMock()
        text_part.function_call = None
        text_part.text = "response text"

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.model_version = "gemini-2.0-flash"

        if thoughts_token_count is not None:
            usage_meta = MagicMock()
            usage_meta.prompt_token_count = 100
            usage_meta.candidates_token_count = 50
            usage_meta.thoughts_token_count = thoughts_token_count
            # Ensure camelCase fallback isn't hit when snake_case is present
            usage_meta.thoughtsTokenCount = None
            usage_meta.cached_content_token_count = None
            usage_meta.cachedContentTokenCount = None
            raw.usage_metadata = usage_meta
        else:
            raw.usage_metadata = None

        return raw

    def test_reasoning_tokens_from_thoughts_token_count(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(thoughts_token_count=350)
        response = adapter._map_response(raw, latency_ms=90.0)
        assert response.usage.reasoning_tokens == 350

    def test_no_usage_metadata_yields_zero(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(thoughts_token_count=None)
        response = adapter._map_response(raw, latency_ms=90.0)
        assert response.usage.reasoning_tokens == 0

    def test_camel_case_fallback(self) -> None:
        """If snake_case attr is missing, fall back to camelCase."""
        adapter = _make_gemini_adapter()
        raw = self._mock_response(thoughts_token_count=0)
        # Override: remove snake_case, set camelCase
        raw.usage_metadata.thoughts_token_count = None
        raw.usage_metadata.thoughtsTokenCount = 275
        response = adapter._map_response(raw, latency_ms=60.0)
        assert response.usage.reasoning_tokens == 275

    def test_input_output_tokens_correct(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(thoughts_token_count=10)
        response = adapter._map_response(raw, latency_ms=50.0)
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50


class TestGeminiCacheReadTokens:
    """L-F01: Verify cache_read_tokens mapped from cachedContentTokenCount."""

    def _mock_response(self, cached_tokens: int = 0):
        text_part = MagicMock()
        text_part.function_call = None
        text_part.text = "response"

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        raw = MagicMock()
        raw.candidates = [candidate]
        raw.model_version = "gemini-2.0-flash"

        usage_meta = MagicMock()
        usage_meta.prompt_token_count = 100
        usage_meta.candidates_token_count = 50
        usage_meta.thoughts_token_count = None
        usage_meta.thoughtsTokenCount = None
        usage_meta.cached_content_token_count = cached_tokens if cached_tokens else None
        usage_meta.cachedContentTokenCount = None
        raw.usage_metadata = usage_meta
        return raw

    def test_cache_read_tokens_mapped(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(cached_tokens=500)
        response = adapter._map_response(raw, latency_ms=40.0)
        assert response.usage.cache_read_tokens == 500

    def test_no_cached_tokens_yields_zero(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(cached_tokens=0)
        response = adapter._map_response(raw, latency_ms=40.0)
        assert response.usage.cache_read_tokens == 0

    def test_camel_case_fallback_for_cache(self) -> None:
        adapter = _make_gemini_adapter()
        raw = self._mock_response(cached_tokens=0)
        raw.usage_metadata.cached_content_token_count = None
        raw.usage_metadata.cachedContentTokenCount = 300
        response = adapter._map_response(raw, latency_ms=40.0)
        assert response.usage.cache_read_tokens == 300


class TestGeminiStreamingTokens:
    """L-F02: Verify streaming path captures reasoning_tokens and cache_read_tokens."""

    @pytest.mark.asyncio
    async def test_streaming_reasoning_tokens(self) -> None:
        adapter = _make_gemini_adapter()

        # Build mock streaming chunks
        text_part = MagicMock()
        text_part.function_call = None
        text_part.text = "streamed text"

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        chunk = MagicMock()
        chunk.candidates = [candidate]

        usage_meta = MagicMock()
        usage_meta.prompt_token_count = 80
        usage_meta.candidates_token_count = 30
        usage_meta.thoughts_token_count = 120
        usage_meta.thoughtsTokenCount = None
        usage_meta.cached_content_token_count = 200
        usage_meta.cachedContentTokenCount = None
        chunk.usage_metadata = usage_meta

        # Mock the async stream
        async def mock_stream():
            yield chunk

        adapter._client.aio.models.generate_content_stream = (
            lambda **kw: _async_return(mock_stream())
        )

        request = _simple_request(model="gemini-2.0-flash")
        events = []
        async for event in adapter.stream(request):
            events.append(event)

        # Find the FINISH event
        finish_events = [e for e in events if e.type.value == "finish"]
        assert len(finish_events) == 1
        usage = finish_events[0].usage
        assert usage is not None
        assert usage.reasoning_tokens == 120
        assert usage.cache_read_tokens == 200
        assert usage.input_tokens == 80
        assert usage.output_tokens == 30

    @pytest.mark.asyncio
    async def test_streaming_no_usage_metadata(self) -> None:
        adapter = _make_gemini_adapter()

        text_part = MagicMock()
        text_part.function_call = None
        text_part.text = "text"

        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        candidate.content.parts = [text_part]

        chunk = MagicMock()
        chunk.candidates = [candidate]
        chunk.usage_metadata = None

        async def mock_stream():
            yield chunk

        adapter._client.aio.models.generate_content_stream = (
            lambda **kw: _async_return(mock_stream())
        )

        request = _simple_request(model="gemini-2.0-flash")
        events = []
        async for event in adapter.stream(request):
            events.append(event)

        finish_events = [e for e in events if e.type.value == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0].usage is None


class TestGeminiSystemInstruction:
    """L-C06: System prompt must use system_instruction, not fake messages."""

    def test_system_prompt_passed_as_system_instruction(self) -> None:
        """System prompt appears in config.system_instruction, not in contents."""
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            system_prompt="You are a helpful coding assistant.",
        )
        kwargs = adapter._build_kwargs(request)

        # system_instruction is set in config
        assert kwargs["config"]["system_instruction"] == "You are a helpful coding assistant."

        # No fake user/model message pair in contents
        contents = kwargs["contents"]
        for entry in contents:
            for part in entry.get("parts", []):
                text = part.get("text", "")
                assert "[System Instructions]" not in text
                assert text != "Understood."

    def test_no_system_prompt_omits_system_instruction(self) -> None:
        """When no system prompt, system_instruction should not appear."""
        adapter = _make_gemini_adapter()
        request = _simple_request(model="gemini-2.0-flash")
        kwargs = adapter._build_kwargs(request)

        config = kwargs.get("config", {})
        assert "system_instruction" not in config

    def test_system_prompt_not_injected_as_messages(self) -> None:
        """Contents should only contain the actual user message, no synthetic pair."""
        adapter = _make_gemini_adapter()
        request = _simple_request(
            model="gemini-2.0-flash",
            system_prompt="Be concise.",
        )
        kwargs = adapter._build_kwargs(request)

        contents = kwargs["contents"]
        # Only the single user message "hello" from _simple_request
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "hello"}]


class TestGeminiFunctionResponseName:
    """L-C07: functionResponse must use function name, not call ID."""

    def test_function_response_uses_function_name(self) -> None:
        """ToolResultContent should map to functionResponse with function name."""
        adapter = _make_gemini_adapter()

        # Simulate a conversation with tool call and tool result
        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolCallContent(
                    tool_call_id="call_abc123",
                    tool_name="read_file",
                    arguments={"path": "/tmp/test.py"},
                ),
            ],
        )
        tool_msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_call_id="call_abc123",
                    content="file contents here",
                ),
            ],
        )
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("hello"), assistant_msg, tool_msg],
        )
        kwargs = adapter._build_kwargs(request)

        # Find the functionResponse part
        tool_result_content = None
        for entry in kwargs["contents"]:
            for part in entry.get("parts", []):
                if "function_response" in part:
                    tool_result_content = part["function_response"]
                    break

        assert tool_result_content is not None
        # Must be the function NAME, not the call ID
        assert tool_result_content["name"] == "read_file"
        assert tool_result_content["name"] != "call_abc123"
        assert tool_result_content["response"] == {"result": "file contents here"}

    def test_multiple_tool_results_use_correct_names(self) -> None:
        """Each tool result maps to its corresponding function name."""
        adapter = _make_gemini_adapter()

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolCallContent(
                    tool_call_id="call_001",
                    tool_name="read_file",
                    arguments={"path": "a.py"},
                ),
                ToolCallContent(
                    tool_call_id="call_002",
                    tool_name="exec_command",
                    arguments={"command": "ls"},
                ),
            ],
        )
        tool_msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(tool_call_id="call_001", content="file a"),
                ToolResultContent(tool_call_id="call_002", content="dir listing"),
            ],
        )
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("do stuff"), assistant_msg, tool_msg],
        )
        kwargs = adapter._build_kwargs(request)

        func_responses = []
        for entry in kwargs["contents"]:
            for part in entry.get("parts", []):
                if "function_response" in part:
                    func_responses.append(part["function_response"])

        assert len(func_responses) == 2
        assert func_responses[0]["name"] == "read_file"
        assert func_responses[0]["response"] == {"result": "file a"}
        assert func_responses[1]["name"] == "exec_command"
        assert func_responses[1]["response"] == {"result": "dir listing"}

    def test_unknown_call_id_falls_back_to_id(self) -> None:
        """If no mapping exists, fall back to the call ID (graceful degradation)."""
        adapter = _make_gemini_adapter()

        # Tool result without a preceding tool call in messages
        tool_msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_call_id="call_unknown",
                    content="some result",
                ),
            ],
        )
        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("hello"), tool_msg],
        )
        kwargs = adapter._build_kwargs(request)

        func_responses = []
        for entry in kwargs["contents"]:
            for part in entry.get("parts", []):
                if "function_response" in part:
                    func_responses.append(part["function_response"])

        assert len(func_responses) == 1
        assert func_responses[0]["name"] == "call_unknown"


class TestGeminiRoundTrip:
    """Round-trip: request with system prompt + tool results → correct Gemini format."""

    def test_full_conversation_format(self) -> None:
        """System prompt + user message + tool call/result produces correct structure."""
        adapter = _make_gemini_adapter()

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolCallContent(
                    tool_call_id="call_rt1",
                    tool_name="grep",
                    arguments={"pattern": "TODO", "path": "."},
                ),
            ],
        )
        tool_msg = Message(
            role=Role.USER,
            content=[
                ToolResultContent(
                    tool_call_id="call_rt1",
                    content="src/main.py:42: # TODO fix this",
                ),
            ],
        )
        request = Request(
            model="gemini-2.0-flash",
            system_prompt="You are a code reviewer.",
            messages=[Message.user("find TODOs"), assistant_msg, tool_msg],
        )
        kwargs = adapter._build_kwargs(request)

        # System instruction in config, not in contents
        assert kwargs["config"]["system_instruction"] == "You are a code reviewer."

        # 3 messages in contents: user, assistant (function_call), user (function_response)
        contents = kwargs["contents"]
        assert len(contents) == 3

        # First: user message
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "find TODOs"}]

        # Second: assistant with function_call
        assert contents[1]["role"] == "model"
        fc = contents[1]["parts"][0]["function_call"]
        assert fc["name"] == "grep"
        assert fc["args"] == {"pattern": "TODO", "path": "."}

        # Third: user with functionResponse using function name
        assert contents[2]["role"] == "user"
        fr = contents[2]["parts"][0]["function_response"]
        assert fr["name"] == "grep"
        assert fr["response"] == {"result": "src/main.py:42: # TODO fix this"}


# ===============================================================
# DEVELOPER role mapping (L-P01)
# ===============================================================


class TestOpenAIDeveloperRole:
    """OpenAI maps DEVELOPER role to 'developer'."""

    def test_developer_role_maps_to_developer(self) -> None:
        adapter = _make_openai_adapter()
        msg = Message(role=Role.DEVELOPER, content=[])
        from attractor.llm.models import TextContent
        msg.content = [TextContent(text="You are a code reviewer.")]
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "developer"
        assert result["content"] == "You are a code reviewer."

    def test_user_role_maps_to_user(self) -> None:
        adapter = _make_openai_adapter()
        msg = Message.user("Hello")
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "user"


class TestAnthropicDeveloperRole:
    """Anthropic maps DEVELOPER role to 'user' (no native developer role)."""

    def test_developer_role_maps_to_user(self) -> None:
        adapter = _make_anthropic_adapter()
        from attractor.llm.models import TextContent
        msg = Message(role=Role.DEVELOPER, content=[TextContent(text="system instructions")])
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "user"

    def test_user_role_maps_to_user(self) -> None:
        adapter = _make_anthropic_adapter()
        msg = Message.user("Hello")
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "user"


class TestGeminiDeveloperRole:
    """Gemini maps DEVELOPER role to 'user' (no native developer role)."""

    def test_developer_role_maps_to_user(self) -> None:
        adapter = _make_gemini_adapter()
        from attractor.llm.models import TextContent
        msg = Message(role=Role.DEVELOPER, content=[TextContent(text="system instructions")])
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "user"

    def test_user_role_maps_to_user(self) -> None:
        adapter = _make_gemini_adapter()
        msg = Message.user("Hello")
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "user"

    def test_assistant_role_maps_to_model(self) -> None:
        adapter = _make_gemini_adapter()
        msg = Message.assistant("Hi there")
        result = adapter._map_message(msg)
        assert result is not None
        assert result["role"] == "model"


# ===============================================================
# Gemini image URL support (L-P03)
# ===============================================================


class TestGeminiImageURLSupport:
    """L-P03: Gemini adapter downloads URL images and converts to base64."""

    @pytest.mark.asyncio
    async def test_url_image_resolved_to_base64(self) -> None:
        """An ImageContent with a URL is downloaded and sent as inline_data."""
        from unittest.mock import AsyncMock, patch

        from attractor.llm.models import ImageContent

        adapter = _make_gemini_adapter()

        png_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
        import base64

        expected_b64 = base64.b64encode(png_bytes).decode("ascii")

        mock_response = AsyncMock()
        mock_response.content = png_bytes
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()

        msg = Message(
            role=Role.USER,
            content=[
                ImageContent(url="https://example.com/image.png"),
            ],
        )
        request = _simple_request(model="gemini-2.0-flash", messages=[msg])

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            resolved = await adapter._resolve_image_urls(request)

        # The resolved request should have base64_data, not url
        img_part = resolved.messages[0].content[0]
        assert isinstance(img_part, ImageContent)
        assert img_part.base64_data == expected_b64
        assert img_part.url is None
        assert img_part.media_type == "image/png"

        # Verify it maps correctly to Gemini format
        kwargs = adapter._build_kwargs(resolved)
        parts = kwargs["contents"][0]["parts"]
        assert len(parts) == 1
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["data"] == expected_b64
        assert parts[0]["inline_data"]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_url_download_failure_skips_image(self) -> None:
        """When URL download fails, the image is gracefully skipped."""
        from unittest.mock import AsyncMock, patch

        from attractor.llm.models import ImageContent, TextContent

        adapter = _make_gemini_adapter()

        msg = Message(
            role=Role.USER,
            content=[
                TextContent(text="Describe this image"),
                ImageContent(url="https://example.com/broken.png"),
            ],
        )
        request = _simple_request(model="gemini-2.0-flash", messages=[msg])

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            resolved = await adapter._resolve_image_urls(request)

        # Original request returned unchanged (download failed)
        assert resolved is request

    @pytest.mark.asyncio
    async def test_base64_image_unchanged(self) -> None:
        """ImageContent with base64_data already set is not re-downloaded."""
        from attractor.llm.models import ImageContent

        adapter = _make_gemini_adapter()

        msg = Message(
            role=Role.USER,
            content=[
                ImageContent(base64_data="aGVsbG8=", media_type="image/jpeg"),
            ],
        )
        request = _simple_request(model="gemini-2.0-flash", messages=[msg])

        # No httpx patching needed — should return request unchanged
        resolved = await adapter._resolve_image_urls(request)

        assert resolved is request
        img = resolved.messages[0].content[0]
        assert isinstance(img, ImageContent)
        assert img.base64_data == "aGVsbG8="

    @pytest.mark.asyncio
    async def test_content_type_from_response_header(self) -> None:
        """Media type is derived from the response Content-Type header."""
        from unittest.mock import AsyncMock, patch

        from attractor.llm.models import ImageContent

        adapter = _make_gemini_adapter()

        mock_response = AsyncMock()
        mock_response.content = b"fake-jpeg"
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        msg = Message(
            role=Role.USER,
            content=[
                ImageContent(url="https://example.com/photo.jpg"),
            ],
        )
        request = _simple_request(model="gemini-2.0-flash", messages=[msg])

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            resolved = await adapter._resolve_image_urls(request)

        img = resolved.messages[0].content[0]
        assert isinstance(img, ImageContent)
        # charset should be stripped
        assert img.media_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_no_images_returns_request_unchanged(self) -> None:
        """Requests without images pass through without modification."""
        adapter = _make_gemini_adapter()
        request = _simple_request(model="gemini-2.0-flash")
        resolved = await adapter._resolve_image_urls(request)
        assert resolved is request
