"""Error hierarchy for the Unified LLM Client.

Defines a structured error hierarchy that maps HTTP status codes and
provider-specific error conditions to typed exceptions with retryability
information. Adapters raise these errors so the retry logic can decide
whether to retry or propagate immediately.
"""

from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base exception for all LLM SDK errors."""

    @property
    def is_retryable(self) -> bool:
        """Whether this error is safe to retry."""
        return False


class ProviderError(SDKError):
    """Base class for errors returned by an LLM provider.

    Attributes:
        provider: Which provider returned the error.
        status_code: HTTP status code, if applicable.
        error_code: Provider-specific error code.
        retryable: Whether this error is safe to retry.
        retry_after: Seconds to wait before retrying.
        raw: Raw error response body from the provider.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw

    @property
    def is_retryable(self) -> bool:
        """Whether this error is safe to retry."""
        return self.retryable


class AuthenticationError(ProviderError):
    """401: Invalid API key or expired token."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class AccessDeniedError(ProviderError):
    """403: Insufficient permissions."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class NotFoundError(ProviderError):
    """404: Model not found, endpoint not found."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class RateLimitError(ProviderError):
    """429: Rate limit exceeded."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class ServerError(ProviderError):
    """500-599: Provider internal error."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class RequestTimeoutError(ProviderError):
    """Request or stream timed out."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class InvalidRequestError(ProviderError):
    """400/422: Malformed request, invalid parameters."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class ContextLengthError(ProviderError):
    """Input + output exceeds context window."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class ContentFilterError(ProviderError):
    """Response blocked by safety filter."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class QuotaExceededError(ProviderError):
    """Billing or usage quota exhausted."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


# ---------------------------------------------------------------------------
# Non-provider SDK errors
# ---------------------------------------------------------------------------


class AbortError(SDKError):
    """Request cancelled via abort signal."""


class NetworkError(SDKError):
    """Network-level failure (connection refused, DNS, etc.)."""

    @property
    def is_retryable(self) -> bool:
        return True


class StreamError(SDKError):
    """Error during stream consumption."""

    @property
    def is_retryable(self) -> bool:
        return True


class InvalidToolCallError(SDKError):
    """Tool call arguments failed validation."""


class NoObjectGeneratedError(SDKError):
    """Structured output parsing or validation failed."""


class ConfigurationError(SDKError):
    """SDK misconfiguration (missing provider, invalid settings, etc.)."""


# ---------------------------------------------------------------------------
# HTTP status code mapping
# ---------------------------------------------------------------------------

_STATUS_TO_ERROR: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AccessDeniedError,
    404: NotFoundError,
    408: RequestTimeoutError,
    413: ContextLengthError,
    422: InvalidRequestError,
    429: RateLimitError,
    500: ServerError,
    502: ServerError,
    503: ServerError,
    504: ServerError,
}


def error_from_status(
    status_code: int,
    message: str,
    *,
    provider: str = "",
    raw: dict[str, Any] | None = None,
    retry_after: float | None = None,
) -> ProviderError:
    """Create the appropriate ProviderError subclass from an HTTP status code.

    Unknown status codes default to a retryable ProviderError (conservative).

    Args:
        status_code: HTTP status code from the provider response.
        message: Error message.
        provider: Provider name.
        raw: Raw error response body.
        retry_after: Seconds to wait before retrying (from Retry-After header).

    Returns:
        An instance of the appropriate ProviderError subclass.
    """
    cls = _STATUS_TO_ERROR.get(status_code, ProviderError)
    kwargs: dict[str, Any] = {
        "provider": provider,
        "status_code": status_code,
        "raw": raw,
        "retry_after": retry_after,
    }
    # Unknown statuses default to retryable
    if cls is ProviderError:
        kwargs["retryable"] = True
    return cls(message, **kwargs)
