"""Tests for SDK error translation across all 3 LLM adapters.

Verifies that raw SDK exceptions from Anthropic, OpenAI, and Gemini
are caught and re-raised as the correct attractor error types per
unified-llm-spec §5.2.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor.llm.errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
)
from attractor.llm.models import Message, Request, Role, TextContent


def _simple_request() -> Request:
    """Create a minimal Request for testing."""
    return Request(
        messages=[Message(role=Role.USER, content=[TextContent(text="hello")])],
        model="test-model",
    )


# ---------------------------------------------------------------------------
# Helper: _extract_retry_after
# ---------------------------------------------------------------------------


class TestExtractRetryAfter:
    """Test the _extract_retry_after helper used by all adapters."""

    def test_extracts_float_from_headers(self) -> None:
        from attractor.llm.adapters.anthropic_adapter import (
            _extract_retry_after,
        )

        exc = MagicMock()
        exc.response.headers = {"retry-after": "2.5"}
        assert _extract_retry_after(exc) == 2.5

    def test_returns_none_when_no_response(self) -> None:
        from attractor.llm.adapters.anthropic_adapter import (
            _extract_retry_after,
        )

        exc = MagicMock(spec=[])  # no response attr
        assert _extract_retry_after(exc) is None

    def test_returns_none_when_no_headers(self) -> None:
        from attractor.llm.adapters.anthropic_adapter import (
            _extract_retry_after,
        )

        exc = MagicMock()
        exc.response.headers = None
        assert _extract_retry_after(exc) is None

    def test_returns_none_for_invalid_value(self) -> None:
        from attractor.llm.adapters.anthropic_adapter import (
            _extract_retry_after,
        )

        exc = MagicMock()
        exc.response.headers = {"retry-after": "not-a-number"}
        assert _extract_retry_after(exc) is None

    def test_returns_none_when_header_missing(self) -> None:
        from attractor.llm.adapters.anthropic_adapter import (
            _extract_retry_after,
        )

        exc = MagicMock()
        exc.response.headers = {}
        assert _extract_retry_after(exc) is None


# ---------------------------------------------------------------------------
# Anthropic error translation
# ---------------------------------------------------------------------------


class TestAnthropicErrorTranslation:
    """Test that AnthropicAdapter.complete() translates SDK exceptions."""

    @pytest.fixture
    def adapter(self) -> Any:
        pytest.importorskip("anthropic")
        from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter

        a = AnthropicAdapter(api_key="test-key")
        return a

    @staticmethod
    def _make_status_error(status_code: int, message: str = "error") -> Any:
        """Create an anthropic.APIStatusError with the given status code."""
        import httpx
        import anthropic

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(status_code, request=request)
        return anthropic.APIStatusError(message, response=response, body={})

    @pytest.mark.asyncio
    async def test_401_becomes_authentication_error(self, adapter: Any) -> None:
        exc = self._make_status_error(401, "invalid api key")
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(AuthenticationError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_429_becomes_rate_limit_error_with_retry_after(
        self, adapter: Any
    ) -> None:
        exc = self._make_status_error(429, "rate limited")
        # Add retry-after header
        exc.response.headers["retry-after"] = "5.0"
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(RateLimitError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.retry_after == 5.0
        assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_500_becomes_server_error(self, adapter: Any) -> None:
        exc = self._make_status_error(500, "internal error")
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(ServerError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_connection_error_becomes_network_error(
        self, adapter: Any
    ) -> None:
        import anthropic
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(request=request)
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(NetworkError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_timeout_error_becomes_request_timeout_error(
        self, adapter: Any
    ) -> None:
        import anthropic
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APITimeoutError(request=request)
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(RequestTimeoutError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_stream_status_error_translated(self, adapter: Any) -> None:
        """Errors during stream creation are also translated."""
        exc = self._make_status_error(401, "unauthorized")
        adapter._client.messages.create = AsyncMock(side_effect=exc)

        with pytest.raises(AuthenticationError):
            async for _ in adapter.stream(_simple_request()):
                pass


# ---------------------------------------------------------------------------
# OpenAI error translation
# ---------------------------------------------------------------------------


class TestOpenAIErrorTranslation:
    """Test that OpenAIAdapter.complete() translates SDK exceptions."""

    @pytest.fixture
    def adapter(self) -> Any:
        pytest.importorskip("openai")
        from attractor.llm.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(api_key="test-key")

    @staticmethod
    def _make_status_error(status_code: int, message: str = "error") -> Any:
        import httpx
        import openai

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(status_code, request=request)
        return openai.APIStatusError(message, response=response, body={})

    @pytest.mark.asyncio
    async def test_401_becomes_authentication_error(self, adapter: Any) -> None:
        exc = self._make_status_error(401)
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(AuthenticationError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.provider == "openai"

    @pytest.mark.asyncio
    async def test_429_becomes_rate_limit_error_with_retry_after(
        self, adapter: Any
    ) -> None:
        exc = self._make_status_error(429, "rate limited")
        exc.response.headers["retry-after"] = "10.0"
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(RateLimitError) as exc_info:
            await adapter.complete(_simple_request())
        assert exc_info.value.retry_after == 10.0

    @pytest.mark.asyncio
    async def test_500_becomes_server_error(self, adapter: Any) -> None:
        exc = self._make_status_error(500)
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(ServerError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_connection_error_becomes_network_error(
        self, adapter: Any
    ) -> None:
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        exc = openai.APIConnectionError(request=request)
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(NetworkError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_timeout_error_becomes_request_timeout_error(
        self, adapter: Any
    ) -> None:
        import openai
        import httpx

        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        exc = openai.APITimeoutError(request=request)
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(RequestTimeoutError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_stream_status_error_translated(self, adapter: Any) -> None:
        exc = self._make_status_error(429, "rate limited")
        adapter._client.responses.create = AsyncMock(side_effect=exc)

        with pytest.raises(RateLimitError):
            async for _ in adapter.stream(_simple_request()):
                pass


# ---------------------------------------------------------------------------
# Gemini error translation
# ---------------------------------------------------------------------------


class TestGeminiErrorTranslation:
    """Test that GeminiAdapter translates SDK exceptions."""

    @pytest.fixture
    def adapter(self) -> Any:
        pytest.importorskip("google.genai")
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(api_key="test-key")

    @pytest.mark.asyncio
    async def test_client_error_400_becomes_invalid_request(
        self, adapter: Any
    ) -> None:
        from google.genai import errors as genai_errors
        from attractor.llm.errors import InvalidRequestError

        exc = genai_errors.ClientError(400, {"error": "bad request"})
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(InvalidRequestError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_client_error_401_becomes_authentication_error(
        self, adapter: Any
    ) -> None:
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(401, {"error": "unauthorized"})
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(AuthenticationError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_client_error_429_becomes_rate_limit_error(
        self, adapter: Any
    ) -> None:
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(429, {"error": "quota exceeded"})
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(RateLimitError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_server_error_500_becomes_server_error(
        self, adapter: Any
    ) -> None:
        from google.genai import errors as genai_errors

        exc = genai_errors.ServerError(500, {"error": "internal error"})
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(ServerError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_timeout_exception_becomes_request_timeout(
        self, adapter: Any
    ) -> None:
        """Name-based detection: exception class name containing 'timeout'."""

        class GeminiTimeoutError(Exception):
            pass

        exc = GeminiTimeoutError("request timed out")
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(RequestTimeoutError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_connection_exception_becomes_network_error(
        self, adapter: Any
    ) -> None:
        """Name-based detection: exception class name containing 'connection'."""

        class GeminiConnectionError(Exception):
            pass

        exc = GeminiConnectionError("connection refused")
        adapter._client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with pytest.raises(NetworkError):
            await adapter.complete(_simple_request())

    @pytest.mark.asyncio
    async def test_stream_error_translated(self, adapter: Any) -> None:
        """Errors during stream creation are translated."""
        from google.genai import errors as genai_errors

        exc = genai_errors.ClientError(401, {"error": "unauthorized"})
        adapter._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=exc
        )

        with pytest.raises(AuthenticationError):
            async for _ in adapter.stream(_simple_request()):
                pass


# ---------------------------------------------------------------------------
# Gemini _translate_error static method
# ---------------------------------------------------------------------------


class TestGeminiTranslateError:
    """Direct tests for the GeminiAdapter._translate_error static method."""

    def test_api_error_with_status_code(self) -> None:
        genai_errors = pytest.importorskip("google.genai.errors")
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter

        exc = genai_errors.APIError(503, {"error": "api error"})

        with pytest.raises(ServerError):
            GeminiAdapter._translate_error(exc)

    def test_unrecognized_exception_does_not_raise(self) -> None:
        """Non-SDK exceptions are not translated — _translate_error returns."""
        from attractor.llm.adapters.gemini_adapter import GeminiAdapter

        exc = ValueError("not an SDK error")
        # Should NOT raise — just returns
        GeminiAdapter._translate_error(exc)
