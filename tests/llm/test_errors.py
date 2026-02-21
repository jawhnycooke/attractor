"""Tests for attractor.llm.errors — error hierarchy, status code mapping, and retry behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from attractor.llm.client import LLMClient
from attractor.llm.errors import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    InvalidToolCallError,
    NetworkError,
    NoObjectGeneratedError,
    NotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    SDKError,
    ServerError,
    StreamError,
    error_from_status,
)
from attractor.llm.models import Message, Request, Response, RetryPolicy


class TestErrorHierarchy:
    def test_provider_error_is_sdk_error(self) -> None:
        err = ProviderError("test")
        assert isinstance(err, SDKError)

    def test_authentication_error_not_retryable(self) -> None:
        err = AuthenticationError("invalid key", provider="anthropic", status_code=401)
        assert err.is_retryable is False
        assert err.provider == "anthropic"
        assert err.status_code == 401

    def test_access_denied_error_not_retryable(self) -> None:
        err = AccessDeniedError("forbidden", provider="openai", status_code=403)
        assert err.is_retryable is False
        assert isinstance(err, ProviderError)

    def test_not_found_error_not_retryable(self) -> None:
        err = NotFoundError("model not found", provider="anthropic", status_code=404)
        assert err.is_retryable is False
        assert isinstance(err, ProviderError)

    def test_rate_limit_error_retryable(self) -> None:
        err = RateLimitError("too many requests", provider="openai", status_code=429)
        assert err.is_retryable is True
        assert err.retry_after is None

    def test_rate_limit_with_retry_after(self) -> None:
        err = RateLimitError(
            "rate limited",
            provider="openai",
            status_code=429,
            retry_after=30.0,
        )
        assert err.retry_after == 30.0

    def test_server_error_retryable(self) -> None:
        err = ServerError("internal error", provider="gemini", status_code=500)
        assert err.is_retryable is True

    def test_timeout_error_retryable(self) -> None:
        err = RequestTimeoutError("timed out", provider="anthropic")
        assert err.is_retryable is True

    def test_invalid_request_not_retryable(self) -> None:
        err = InvalidRequestError("bad params", provider="openai", status_code=400)
        assert err.is_retryable is False

    def test_context_length_not_retryable(self) -> None:
        err = ContextLengthError("too long", provider="anthropic", status_code=413)
        assert err.is_retryable is False

    def test_content_filter_not_retryable(self) -> None:
        err = ContentFilterError("blocked", provider="openai")
        assert err.is_retryable is False

    def test_quota_exceeded_not_retryable(self) -> None:
        err = QuotaExceededError("quota used", provider="openai")
        assert err.is_retryable is False
        assert isinstance(err, ProviderError)

    def test_provider_error_raw_field(self) -> None:
        err = ProviderError(
            "test",
            provider="mock",
            raw={"error": {"type": "test"}},
        )
        assert err.raw == {"error": {"type": "test"}}

    def test_all_provider_subclasses(self) -> None:
        subclasses = [
            AuthenticationError,
            AccessDeniedError,
            NotFoundError,
            RateLimitError,
            ServerError,
            RequestTimeoutError,
            InvalidRequestError,
            ContextLengthError,
            ContentFilterError,
            QuotaExceededError,
        ]
        for cls in subclasses:
            err = cls("test")
            assert isinstance(err, ProviderError)
            assert isinstance(err, SDKError)


class TestNonProviderErrors:
    """Tests for SDKError subclasses that are NOT ProviderErrors."""

    def test_abort_error_not_retryable(self) -> None:
        err = AbortError("cancelled")
        assert err.is_retryable is False
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_network_error_retryable(self) -> None:
        err = NetworkError("connection refused")
        assert err.is_retryable is True
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_stream_error_retryable(self) -> None:
        err = StreamError("stream interrupted")
        assert err.is_retryable is True
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_invalid_tool_call_error_not_retryable(self) -> None:
        err = InvalidToolCallError("bad args")
        assert err.is_retryable is False
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_no_object_generated_error_not_retryable(self) -> None:
        err = NoObjectGeneratedError("parse failed")
        assert err.is_retryable is False
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_configuration_error_not_retryable(self) -> None:
        err = ConfigurationError("missing provider")
        assert err.is_retryable is False
        assert isinstance(err, SDKError)
        assert not isinstance(err, ProviderError)

    def test_sdk_error_base_not_retryable(self) -> None:
        err = SDKError("generic")
        assert err.is_retryable is False


class TestErrorFromStatus:
    def test_401_maps_to_authentication_error(self) -> None:
        err = error_from_status(401, "unauthorized", provider="anthropic")
        assert isinstance(err, AuthenticationError)
        assert err.is_retryable is False

    def test_403_maps_to_access_denied_error(self) -> None:
        err = error_from_status(403, "forbidden", provider="openai")
        assert isinstance(err, AccessDeniedError)
        assert err.is_retryable is False

    def test_404_maps_to_not_found_error(self) -> None:
        err = error_from_status(404, "not found", provider="anthropic")
        assert isinstance(err, NotFoundError)
        assert err.is_retryable is False

    def test_400_maps_to_invalid_request(self) -> None:
        err = error_from_status(400, "bad request", provider="openai")
        assert isinstance(err, InvalidRequestError)
        assert err.is_retryable is False

    def test_429_maps_to_rate_limit(self) -> None:
        err = error_from_status(
            429, "rate limited", provider="openai", retry_after=5.0
        )
        assert isinstance(err, RateLimitError)
        assert err.is_retryable is True
        assert err.retry_after == 5.0

    def test_500_maps_to_server_error(self) -> None:
        err = error_from_status(500, "internal error", provider="anthropic")
        assert isinstance(err, ServerError)
        assert err.is_retryable is True

    def test_502_maps_to_server_error(self) -> None:
        err = error_from_status(502, "bad gateway", provider="gemini")
        assert isinstance(err, ServerError)

    def test_413_maps_to_context_length(self) -> None:
        err = error_from_status(413, "too large", provider="openai")
        assert isinstance(err, ContextLengthError)
        assert err.is_retryable is False

    def test_unknown_status_defaults_retryable(self) -> None:
        err = error_from_status(418, "i'm a teapot", provider="mock")
        assert isinstance(err, ProviderError)
        assert err.is_retryable is True

    def test_raw_field_preserved(self) -> None:
        raw = {"error": {"message": "test", "type": "invalid_request"}}
        err = error_from_status(400, "bad request", provider="openai", raw=raw)
        assert err.raw == raw


# ---------------------------------------------------------------------------
# Behavioral tests: verify _complete_with_retry actually respects is_retryable
# ---------------------------------------------------------------------------


def _make_client_and_adapter(
    retry_policy: RetryPolicy | None = None,
) -> tuple[LLMClient, AsyncMock]:
    """Create an LLMClient with a mock adapter for retry behavior testing."""
    adapter = AsyncMock()
    adapter.detect_model.return_value = True
    adapter.provider_name.return_value = "mock"
    policy = retry_policy or RetryPolicy(max_retries=2, base_delay_seconds=0.001)
    client = LLMClient(adapters=[adapter], retry_policy=policy)
    return client, adapter


def _make_request() -> Request:
    return Request(messages=[Message.user("hello")], model="mock-model")


def _make_response() -> Response:
    return Response(
        message=Message.assistant("ok"),
        model="mock-model",
    )


class TestRetryBehaviorNonRetryable:
    """Verify that _complete_with_retry does NOT retry non-retryable errors."""

    @pytest.mark.asyncio
    async def test_authentication_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = AuthenticationError(
            "invalid key", provider="mock"
        )
        with pytest.raises(AuthenticationError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_access_denied_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = AccessDeniedError(
            "forbidden", provider="mock"
        )
        with pytest.raises(AccessDeniedError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_not_found_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = NotFoundError(
            "model not found", provider="mock"
        )
        with pytest.raises(NotFoundError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_request_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = InvalidRequestError(
            "bad params", provider="mock"
        )
        with pytest.raises(InvalidRequestError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_context_length_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = ContextLengthError(
            "too long", provider="mock"
        )
        with pytest.raises(ContextLengthError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_content_filter_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = ContentFilterError(
            "blocked", provider="mock"
        )
        with pytest.raises(ContentFilterError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_quota_exceeded_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = QuotaExceededError(
            "quota used", provider="mock"
        )
        with pytest.raises(QuotaExceededError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_abort_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = AbortError("cancelled")
        with pytest.raises(AbortError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_tool_call_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = InvalidToolCallError("bad args")
        with pytest.raises(InvalidToolCallError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_no_object_generated_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = NoObjectGeneratedError("parse failed")
        with pytest.raises(NoObjectGeneratedError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_configuration_error_no_retry(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = ConfigurationError("bad config")
        with pytest.raises(ConfigurationError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 1


class TestRetryBehaviorRetryable:
    """Verify that _complete_with_retry DOES retry retryable errors."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_retried(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = RateLimitError(
            "rate limited", provider="mock"
        )
        with pytest.raises(RateLimitError):
            await client._complete_with_retry(adapter, _make_request())
        # 1 initial + 2 retries = 3 total calls
        assert adapter.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_server_error_retried(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = ServerError(
            "internal error", provider="mock"
        )
        with pytest.raises(ServerError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error_retried(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = RequestTimeoutError(
            "timed out", provider="mock"
        )
        with pytest.raises(RequestTimeoutError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_network_error_retried(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = NetworkError("connection refused")
        with pytest.raises(NetworkError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_stream_error_retried(self) -> None:
        client, adapter = _make_client_and_adapter()
        adapter.complete.side_effect = StreamError("stream broken")
        with pytest.raises(StreamError):
            await client._complete_with_retry(adapter, _make_request())
        assert adapter.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_retryable_error_succeeds_on_retry(self) -> None:
        """Verify a transient error followed by success returns the response."""
        client, adapter = _make_client_and_adapter()
        resp = _make_response()
        adapter.complete.side_effect = [
            ServerError("internal error", provider="mock"),
            resp,
        ]
        result = await client._complete_with_retry(adapter, _make_request())
        assert result is resp
        assert adapter.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_delay_used(self) -> None:
        """Verify that retry_after from the error is used as delay."""
        client, adapter = _make_client_and_adapter()
        resp = _make_response()
        adapter.complete.side_effect = [
            RateLimitError("rate limited", provider="mock", retry_after=5.0),
            resp,
        ]
        with patch("attractor.llm.client.asyncio.sleep") as mock_sleep:
            result = await client._complete_with_retry(adapter, _make_request())
            assert result is resp
            mock_sleep.assert_called_once_with(5.0)
