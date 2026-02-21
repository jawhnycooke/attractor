"""Base protocol for LLM provider adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from attractor.llm.models import Request, Response, StreamEvent


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol that all provider adapters must satisfy.

    Each adapter translates between the provider-agnostic Request/Response
    models and a specific LLM provider's API format.
    """

    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'openai', 'anthropic')."""
        ...

    def detect_model(self, model: str) -> bool:
        """Return True if this adapter handles the given model string."""
        ...

    async def complete(self, request: Request) -> Response:
        """Send a non-streaming completion request."""
        ...

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request, yielding events as they arrive."""
        ...
