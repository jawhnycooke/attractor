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
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


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
