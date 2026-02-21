"""Google Gemini provider adapter."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from attractor.llm.models import (
    ContentPart,
    FinishReason,
    ImageContent,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_FINISH_MAP = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
}


def _synth_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


class GeminiAdapter:
    """Adapter for the Google Gemini (google-genai) API."""

    def __init__(self, api_key: str | None = None) -> None:
        from google import genai

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
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

    def _map_contents(self, request: Request) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []

        if request.system_prompt:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {"text": f"[System Instructions]\n{request.system_prompt}"}
                    ],
                }
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [{"text": "Understood."}],
                }
            )

        for msg in request.messages:
            mapped = self._map_message(msg)
            if mapped is not None:
                contents.append(mapped)

        return contents

    def _map_message(self, msg: Message) -> dict[str, Any] | None:
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
                parts.append(
                    {
                        "function_response": {
                            "name": part.tool_call_id,
                            "response": {"result": part.content},
                        },
                    }
                )

        role = "model" if msg.role == Role.ASSISTANT else "user"
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

    def _build_kwargs(self, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "contents": self._map_contents(request),
        }
        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop_sequences:
            config["stop_sequences"] = request.stop_sequences

        tools = self._map_tools(request.tools)
        if tools:
            config["tools"] = tools

        if config:
            kwargs["config"] = config

        return kwargs

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
            usage = TokenUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            )

        finish = _FINISH_MAP.get(finish_str, FinishReason.STOP)
        if content_parts and any(isinstance(p, ToolCallContent) for p in content_parts):
            finish = FinishReason.TOOL_USE

        return Response(
            message=Message(role=Role.ASSISTANT, content=content_parts),
            model=raw.model_version if hasattr(raw, "model_version") else "",
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
        """
        kwargs = self._build_kwargs(request)
        t0 = time.monotonic()
        raw = await self._client.aio.models.generate_content(**kwargs)
        latency = (time.monotonic() - t0) * 1000
        return self._map_response(raw, latency)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request to the Google Gemini API.

        Args:
            request: Provider-agnostic request to send.

        Yields:
            StreamEvent instances as they arrive from the API.
        """
        kwargs = self._build_kwargs(request)

        t0 = time.monotonic()
        raw_stream = await self._client.aio.models.generate_content_stream(**kwargs)

        yield StreamEvent(type=StreamEventType.STREAM_START)

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
                    usage = TokenUsage(
                        input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage_meta, "candidates_token_count", 0)
                        or 0,
                    )
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=_FINISH_MAP.get(fr_str, FinishReason.STOP),
                    usage=usage,
                    metadata={"latency_ms": (time.monotonic() - t0) * 1000},
                )
