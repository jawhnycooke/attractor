"""Google Gemini provider adapter."""

from __future__ import annotations

import base64 as base64_mod
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx

from attractor.llm.errors import (
    NetworkError,
    RequestTimeoutError,
    StreamError,
    error_from_status,
)
from attractor.llm.models import (
    ContentPart,
    FinishReason,
    ImageContent,
    Message,
    ReasoningEffort,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCallContent,
    ToolChoice,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_THINKING_BUDGET = {
    ReasoningEffort.LOW: 2048,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 32768,
}

_FINISH_MAP: dict[str, FinishReason] = {
    "STOP": FinishReason("stop", raw="STOP"),
    "MAX_TOKENS": FinishReason("length", raw="MAX_TOKENS"),
    "SAFETY": FinishReason("content_filter", raw="SAFETY"),
    "RECITATION": FinishReason("content_filter", raw="RECITATION"),
}


def _synth_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _extract_retry_after(exc: Any) -> float | None:
    """Extract Retry-After from a Gemini SDK exception."""
    # Gemini SDK exceptions may have different structures
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_str = headers.get("retry-after")
    if retry_str is None:
        return None
    try:
        return float(retry_str)
    except (ValueError, TypeError):
        return None


class GeminiAdapter:
    """Adapter for the Google Gemini (google-genai) API."""

    def __init__(self, api_key: str | None = None) -> None:
        from google import genai

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini adapter"
            )
        self._client = genai.Client(api_key=self._api_key)

    def provider_name(self) -> str:
        """Return the canonical provider name.

        Returns:
            The string ``"gemini"``.
        """
        return "gemini"

    def detect_model(self, model: str) -> bool:
        """Check whether *model* belongs to this provider.

        Args:
            model: Model identifier string (e.g. ``"gemini-2.0-flash"``).

        Returns:
            True if the model string starts with ``"gemini-"``.
        """
        return model.startswith("gemini-")

    # -----------------------------------------------------------------
    # Request mapping
    # -----------------------------------------------------------------

    def _build_tool_call_id_map(self, messages: list[Message]) -> dict[str, str]:
        """Build a mapping of tool_call_id → tool_name from messages.

        Gemini's ``functionResponse`` requires the function name, not the
        tool call ID.  This scans assistant messages for ``ToolCallContent``
        parts and records the mapping so that ``_map_message`` can look up
        the correct function name for each ``ToolResultContent``.
        """
        mapping: dict[str, str] = {}
        for msg in messages:
            for part in msg.content:
                if isinstance(part, ToolCallContent):
                    mapping[part.tool_call_id] = part.tool_name
        return mapping

    def _map_contents(
        self, request: Request
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert request messages to Gemini contents format.

        Returns:
            A tuple of (contents list, system_instruction or None).
        """
        contents: list[dict[str, Any]] = []
        tool_id_map = self._build_tool_call_id_map(request.messages)

        for msg in request.messages:
            mapped = self._map_message(msg, tool_id_map)
            if mapped is not None:
                contents.append(mapped)

        system_instruction = request.system_prompt or None
        return contents, system_instruction

    def _map_message(
        self,
        msg: Message,
        tool_id_map: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        if msg.role == Role.SYSTEM:
            return None

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, TextContent):
                parts.append({"text": part.text})
            elif isinstance(part, ImageContent):
                if part.base64_data:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": part.media_type,
                                "data": part.base64_data,
                            },
                        }
                    )
            elif isinstance(part, ToolCallContent):
                args = part.arguments or {}
                parts.append(
                    {
                        "function_call": {
                            "name": part.tool_name,
                            "args": args,
                        },
                    }
                )
            elif isinstance(part, ToolResultContent):
                # Gemini functionResponse uses the function NAME,
                # not the tool call ID (spec §7.5).
                func_name = (
                    (tool_id_map or {}).get(part.tool_call_id)
                    or part.tool_call_id
                )
                parts.append(
                    {
                        "function_response": {
                            "name": func_name,
                            "response": {"result": part.content},
                        },
                    }
                )

        if msg.role == Role.ASSISTANT:
            role = "model"
        elif msg.role == Role.DEVELOPER:
            # Gemini has no developer role; map to user
            role = "user"
        else:
            role = "user"
        return {"role": role, "parts": parts} if parts else None

    def _map_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        declarations = []
        for t in tools:
            decl: dict[str, Any] = {
                "name": t.name,
                "description": t.description,
            }
            schema = t.to_json_schema()
            if schema.get("properties"):
                decl["parameters"] = schema
            declarations.append(decl)
        return [{"function_declarations": declarations}]

    async def _resolve_image_urls(self, request: Request) -> Request:
        """Download URL-based images and convert to base64 inline data.

        The Gemini API does not accept arbitrary HTTP image URLs.  This
        method scans the request messages for ``ImageContent`` parts that
        carry a ``url`` but no ``base64_data``, downloads them via
        ``httpx``, and returns a **new** ``Request`` with those parts
        replaced by base64-encoded equivalents.

        Args:
            request: The original provider-agnostic request.

        Returns:
            A (possibly new) Request with URL images resolved to base64.
        """
        # Collect unique URLs that need fetching
        urls: set[str] = set()
        for msg in request.messages:
            for part in msg.content:
                if isinstance(part, ImageContent) and part.url and not part.base64_data:
                    urls.add(part.url)

        if not urls:
            return request

        # Download each URL
        url_data: dict[str, tuple[str, str]] = {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    b64 = base64_mod.b64encode(resp.content).decode("ascii")
                    content_type = (
                        resp.headers.get("content-type", "image/png").split(";")[0]
                    )
                    url_data[url] = (b64, content_type)
                except Exception as exc:
                    logger.warning("Failed to download image from %s: %s", url, exc)

        if not url_data:
            return request

        # Rebuild messages, replacing resolved ImageContent parts
        new_messages: list[Message] = []
        for msg in request.messages:
            new_parts: list[ContentPart] = []
            changed = False
            for part in msg.content:
                if (
                    isinstance(part, ImageContent)
                    and part.url
                    and part.url in url_data
                ):
                    b64, media_type = url_data[part.url]
                    new_parts.append(
                        ImageContent(
                            base64_data=b64,
                            media_type=media_type,
                            detail=part.detail,
                        )
                    )
                    changed = True
                else:
                    new_parts.append(part)
            if changed:
                new_messages.append(Message(role=msg.role, content=new_parts))
            else:
                new_messages.append(msg)

        return replace(request, messages=new_messages)

    def _build_kwargs(self, request: Request) -> dict[str, Any]:
        contents, system_instruction = self._map_contents(request)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "contents": contents,
        }
        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop_sequences:
            config["stop_sequences"] = request.stop_sequences

        # Structured output (response_format → response_mime_type + response_schema)
        if request.response_format:
            json_schema_block = request.response_format.get("json_schema", {})
            schema = json_schema_block.get("schema", {})
            if schema:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = self._strip_unsupported_schema_keys(schema)

        # Reasoning effort → thinking_config
        if request.reasoning_effort and "thinking_config" not in config:
            budget = _THINKING_BUDGET.get(request.reasoning_effort, 8192)
            config["thinking_config"] = {"thinking_budget": budget}

        tools = self._map_tools(request.tools)
        if tools:
            config["tools"] = tools

        # Tool choice mapping
        if request.tool_choice and tools:
            tc_mapped = self._map_tool_choice(request.tool_choice)
            if tc_mapped is not None:
                if tc_mapped == "__none__":
                    config.pop("tools", None)
                else:
                    config["tool_config"] = {"function_calling_config": tc_mapped}

        # Read provider_options
        if request.provider_options:
            gemini_opts = request.provider_options.get("gemini", {})
            if isinstance(gemini_opts, dict):
                # Safety settings
                safety_settings = gemini_opts.get("safety_settings")
                if safety_settings:
                    config["safety_settings"] = safety_settings
                # Thinking config
                thinking_config = gemini_opts.get("thinking_config")
                if thinking_config:
                    config["thinking_config"] = thinking_config
                # Cached content
                cached_content = gemini_opts.get("cached_content")
                if cached_content:
                    kwargs["cached_content"] = cached_content

        if config:
            kwargs["config"] = config

        return kwargs

    @staticmethod
    def _strip_unsupported_schema_keys(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove JSON Schema keys not supported by the Gemini API.

        Gemini rejects ``additionalProperties`` (converted to
        ``additional_properties`` internally). This method recursively
        strips such keys from the schema and any nested sub-schemas.
        """
        _UNSUPPORTED = {"additionalProperties"}
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key in _UNSUPPORTED:
                continue
            if isinstance(value, dict):
                cleaned[key] = GeminiAdapter._strip_unsupported_schema_keys(value)
            elif key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    k: GeminiAdapter._strip_unsupported_schema_keys(v)
                    if isinstance(v, dict) else v
                    for k, v in value.items()
                }
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _map_tool_choice(
        choice: ToolChoice | str | dict[str, Any] | None,
    ) -> dict[str, Any] | str | None:
        """Map a unified ToolChoice to Gemini's tool_config format.

        Returns ``"__none__"`` sentinel when tools should be removed.
        """
        if choice is None:
            return None

        if isinstance(choice, str):
            choice = ToolChoice(mode=choice)
        elif isinstance(choice, dict):
            choice = ToolChoice(
                mode=choice.get("mode", "auto"),
                tool_name=choice.get("tool_name"),
            )

        if choice.mode == "auto":
            return {"mode": "AUTO"}
        elif choice.mode == "none":
            return "__none__"
        elif choice.mode == "required":
            return {"mode": "ANY"}
        elif choice.mode == "named":
            return {
                "mode": "ANY",
                "allowed_function_names": [choice.tool_name],
            }
        return None

    # -----------------------------------------------------------------
    # Response mapping
    # -----------------------------------------------------------------

    def _map_response(self, raw: Any, latency_ms: float) -> Response:
        content_parts: list[ContentPart] = []

        candidate = raw.candidates[0] if raw.candidates else None
        finish_str = "STOP"

        if candidate:
            finish_str = getattr(candidate, "finish_reason", "STOP") or "STOP"
            if isinstance(finish_str, int):
                finish_str = "STOP"
            else:
                finish_str = str(finish_str).split(".")[-1]  # handle enum values

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    content_parts.append(
                        ToolCallContent(
                            tool_call_id=_synth_tool_call_id(),
                            tool_name=fc.name,
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                    )
                elif hasattr(part, "text") and part.text:
                    content_parts.append(TextContent(text=part.text))

        usage = TokenUsage()
        usage_meta = getattr(raw, "usage_metadata", None)
        if usage_meta:
            # Try both camelCase and snake_case for the thoughts token field
            thoughts_tokens = (
                getattr(usage_meta, "thoughts_token_count", None)
                or getattr(usage_meta, "thoughtsTokenCount", None)
                or 0
            )
            cache_read_tokens = (
                getattr(usage_meta, "cached_content_token_count", None)
                or getattr(usage_meta, "cachedContentTokenCount", None)
                or 0
            )
            usage = TokenUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
                reasoning_tokens=thoughts_tokens,
                cache_read_tokens=cache_read_tokens,
            )

        finish = _FINISH_MAP.get(finish_str, FinishReason("stop", raw=finish_str))
        if content_parts and any(isinstance(p, ToolCallContent) for p in content_parts):
            finish = FinishReason("tool_calls", raw=finish_str)

        return Response(
            message=Message(role=Role.ASSISTANT, content=content_parts),
            model=raw.model_version if hasattr(raw, "model_version") else "",
            provider="gemini",
            finish_reason=finish,
            usage=usage,
            latency_ms=latency_ms,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        """Send a non-streaming request to the Google Gemini API.

        Args:
            request: Provider-agnostic request to send.

        Returns:
            Mapped Response with content, usage, and finish reason.

        Raises:
            ProviderError: Translated from raw Gemini SDK exceptions.
        """
        request = await self._resolve_image_urls(request)
        kwargs = self._build_kwargs(request)
        t0 = time.monotonic()
        try:
            raw = await self._client.aio.models.generate_content(**kwargs)
        except Exception as exc:
            self._translate_error(exc)
            raise  # unreachable if _translate_error raises
        latency = (time.monotonic() - t0) * 1000
        return self._map_response(raw, latency)

    @staticmethod
    def _translate_error(exc: Exception) -> None:
        """Translate Gemini SDK exceptions to attractor error types.

        Raises the translated error, or returns if the exception
        isn't a recognized Gemini error (allowing re-raise by caller).
        """
        try:
            from google.genai import errors as genai_errors
        except ImportError:
            return  # Can't translate without SDK

        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

        if isinstance(exc, genai_errors.ClientError):
            code = status_code or 400
            retry_after = _extract_retry_after(exc)
            raise error_from_status(
                code, str(exc), provider="gemini", retry_after=retry_after
            ) from exc
        elif isinstance(exc, genai_errors.ServerError):
            code = status_code or 500
            retry_after = _extract_retry_after(exc)
            raise error_from_status(
                code, str(exc), provider="gemini", retry_after=retry_after
            ) from exc
        elif isinstance(exc, genai_errors.APIError):
            code = status_code or 500
            retry_after = _extract_retry_after(exc)
            raise error_from_status(
                code, str(exc), provider="gemini", retry_after=retry_after
            ) from exc

        # Check for connection/timeout errors via exception name/type
        exc_name = type(exc).__name__.lower()
        if "timeout" in exc_name:
            raise RequestTimeoutError(
                str(exc), provider="gemini", retryable=True
            ) from exc
        if "connection" in exc_name:
            raise NetworkError(str(exc)) from exc

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request to the Google Gemini API.

        Args:
            request: Provider-agnostic request to send.

        Yields:
            StreamEvent instances as they arrive from the API.

        Raises:
            ProviderError: Translated from raw Gemini SDK exceptions.
        """
        request = await self._resolve_image_urls(request)
        kwargs = self._build_kwargs(request)

        t0 = time.monotonic()
        try:
            raw_stream = await self._client.aio.models.generate_content_stream(**kwargs)
        except Exception as exc:
            self._translate_error(exc)
            raise

        yield StreamEvent(type=StreamEventType.STREAM_START)

        try:
            async for chunk in raw_stream:
                candidate = chunk.candidates[0] if chunk.candidates else None
                if not candidate:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        args = dict(fc.args) if fc.args else {}
                        tc = ToolCallContent(
                            tool_call_id=_synth_tool_call_id(),
                            tool_name=fc.name,
                            arguments=args,
                            arguments_json=json.dumps(args),
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call=tc,
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_END,
                            tool_call=tc,
                        )
                    elif hasattr(part, "text") and part.text:
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            text=part.text,
                        )

                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason:
                    fr_str = str(finish_reason).split(".")[-1]
                    usage_meta = getattr(chunk, "usage_metadata", None)
                    usage = None
                    if usage_meta:
                        thoughts_tokens = (
                            getattr(usage_meta, "thoughts_token_count", None)
                            or getattr(usage_meta, "thoughtsTokenCount", None)
                            or 0
                        )
                        cache_read_tokens = (
                            getattr(usage_meta, "cached_content_token_count", None)
                            or getattr(usage_meta, "cachedContentTokenCount", None)
                            or 0
                        )
                        usage = TokenUsage(
                            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                            output_tokens=getattr(usage_meta, "candidates_token_count", 0)
                            or 0,
                            reasoning_tokens=thoughts_tokens,
                            cache_read_tokens=cache_read_tokens,
                        )
                    yield StreamEvent(
                        type=StreamEventType.FINISH,
                        finish_reason=_FINISH_MAP.get(fr_str, FinishReason("stop", raw=fr_str)),
                        usage=usage,
                        metadata={"latency_ms": (time.monotonic() - t0) * 1000},
                    )
        except Exception as exc:
            self._translate_error(exc)
            raise
