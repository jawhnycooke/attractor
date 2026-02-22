"""OpenAI provider adapter using the Responses API."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

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
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolChoice,
    ToolDefinition,
    ToolResultContent,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def _extract_retry_after(exc: Any) -> float | None:
    """Extract Retry-After header from an SDK exception's response."""
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


class OpenAIAdapter:
    """Adapter for the OpenAI Responses API.

    Uses ``client.responses.create()`` — the newer Responses API — which
    returns typed output items (message, function_call, reasoning) rather
    than the older ``chat.completions`` choices array.
    """

    _MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "codex-")

    def __init__(self, api_key: str | None = None) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def provider_name(self) -> str:
        """Return the canonical provider name.

        Returns:
            The string ``"openai"``.
        """
        return "openai"

    def detect_model(self, model: str) -> bool:
        """Check whether *model* belongs to this provider.

        Args:
            model: Model identifier string (e.g. ``"gpt-4o"``).

        Returns:
            True if the model string starts with a known OpenAI prefix.
        """
        return any(model.startswith(p) for p in self._MODEL_PREFIXES)

    # -----------------------------------------------------------------
    # Request mapping
    # -----------------------------------------------------------------

    def _map_input(self, request: Request) -> list[dict[str, Any]] | str:
        """Map our Request messages into Responses API input items."""
        items: list[dict[str, Any]] = []

        for msg in request.messages:
            mapped = self._map_message(msg)
            if mapped is not None:
                if isinstance(mapped, list):
                    items.extend(mapped)
                else:
                    items.append(mapped)

        if len(items) == 1 and items[0].get("role") == "user":
            content = items[0].get("content", "")
            if isinstance(content, str):
                return content

        return items

    def _map_message(
        self, msg: Message
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if msg.role == Role.SYSTEM:
            return None  # system is handled via instructions parameter

        if msg.role == Role.TOOL:
            results = []
            for part in msg.content:
                if isinstance(part, ToolResultContent):
                    results.append(
                        {
                            "type": "function_call_output",
                            "call_id": part.tool_call_id,
                            "output": part.content,
                        }
                    )
            return results if len(results) != 1 else results[0]

        if msg.role == Role.ASSISTANT:
            items: list[dict[str, Any]] = []
            text_parts = []

            for part in msg.content:
                if isinstance(part, TextContent):
                    text_parts.append(part.text)
                elif isinstance(part, ToolCallContent):
                    args_str = part.arguments_json or json.dumps(part.arguments)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": part.tool_call_id,
                            "name": part.tool_name,
                            "arguments": args_str,
                        }
                    )

            if text_parts and not items:
                return {
                    "role": "assistant",
                    "content": "".join(text_parts),
                }
            if text_parts:
                items.insert(
                    0,
                    {
                        "role": "assistant",
                        "content": "".join(text_parts),
                    },
                )
            return items if len(items) != 1 else items[0]

        # User / developer messages
        content = self._map_content_parts(msg.content)
        if msg.role == Role.DEVELOPER:
            role = "developer"
        else:
            role = "user"
        return {"role": role, "content": content}

    def _map_content_parts(
        self, parts: list[ContentPart]
    ) -> str | list[dict[str, Any]]:
        if len(parts) == 1 and isinstance(parts[0], TextContent):
            return parts[0].text

        mapped: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextContent):
                mapped.append({"type": "input_text", "text": part.text})
            elif isinstance(part, ImageContent):
                url = part.url or f"data:{part.media_type};base64,{part.base64_data}"
                mapped.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                    }
                )
        return mapped

    def _map_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.to_json_schema(),
                **({"strict": True} if t.strict else {}),
            }
            for t in tools
        ]

    def _build_kwargs(self, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": self._map_input(request),
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt

        tools = self._map_tools(request.tools)
        if tools:
            kwargs["tools"] = tools

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.reasoning_effort.value}

        if request.response_format:
            kwargs["text"] = {"format": request.response_format}

        # Tool choice mapping
        if request.tool_choice and kwargs.get("tools"):
            tc_mapped = self._map_tool_choice(request.tool_choice)
            if tc_mapped is not None:
                if tc_mapped == "__none__":
                    kwargs.pop("tools", None)
                else:
                    kwargs["tool_choice"] = tc_mapped

        # Read provider_options
        if request.provider_options:
            openai_opts = request.provider_options.get("openai", {})
            if isinstance(openai_opts, dict):
                # Extra body params merged into kwargs
                extra_body = openai_opts.get("extra_body", {})
                if extra_body and isinstance(extra_body, dict):
                    kwargs.update(extra_body)
                # Extra headers stored for the create() call
                extra_headers = openai_opts.get("extra_headers", {})
                if extra_headers:
                    kwargs["extra_headers"] = extra_headers

        return kwargs

    @staticmethod
    def _map_tool_choice(
        choice: ToolChoice | str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        """Map a unified ToolChoice to OpenAI's tool_choice format.

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
            return "auto"
        elif choice.mode == "none":
            return "__none__"
        elif choice.mode == "required":
            return "required"
        elif choice.mode == "named":
            return {
                "type": "function",
                "function": {"name": choice.tool_name},
            }
        return None

    # -----------------------------------------------------------------
    # Response mapping
    # -----------------------------------------------------------------

    def _map_response(self, raw: Any, latency_ms: float) -> Response:
        content_parts: list[ContentPart] = []
        has_tool_calls = False

        for item in raw.output:
            item_type = item.type

            if item_type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text":
                        content_parts.append(TextContent(text=content_block.text))
                    elif content_block.type == "refusal":
                        content_parts.append(
                            TextContent(text=f"[Refusal: {content_block.refusal}]")
                        )

            elif item_type == "function_call":
                has_tool_calls = True
                args_str = item.arguments
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to parse tool call arguments: %s", exc)
                    args = {}
                content_parts.append(
                    ToolCallContent(
                        tool_call_id=item.call_id,
                        tool_name=item.name,
                        arguments=args,
                        arguments_json=args_str,
                    )
                )

            elif item_type == "reasoning":
                for summary_item in getattr(item, "summary", []):
                    if hasattr(summary_item, "text"):
                        content_parts.append(ThinkingContent(text=summary_item.text))

        usage = TokenUsage()
        if raw.usage:
            reasoning_tokens = 0
            cache_read_tokens = 0
            if raw.usage.output_tokens_details:
                reasoning_tokens = (
                    getattr(raw.usage.output_tokens_details, "reasoning_tokens", 0) or 0
                )
            if raw.usage.input_tokens_details:
                cache_read_tokens = (
                    getattr(raw.usage.input_tokens_details, "cached_tokens", 0) or 0
                )
            usage = TokenUsage(
                input_tokens=raw.usage.input_tokens or 0,
                output_tokens=raw.usage.output_tokens or 0,
                reasoning_tokens=reasoning_tokens,
                cache_read_tokens=cache_read_tokens,
            )

        if has_tool_calls:
            finish = FinishReason("tool_calls", raw="tool_calls")
        elif raw.status == "incomplete":
            finish = FinishReason("length", raw="incomplete")
        else:
            finish = FinishReason("stop", raw=raw.status)

        return Response(
            message=Message(role=Role.ASSISTANT, content=content_parts),
            model=raw.model if isinstance(raw.model, str) else str(raw.model),
            provider="openai",
            finish_reason=finish,
            usage=usage,
            provider_response_id=raw.id or "",
            latency_ms=latency_ms,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        """Send a non-streaming completion request to the OpenAI Responses API.

        Args:
            request: Provider-agnostic request to send.

        Returns:
            Mapped Response with content, usage, and finish reason.

        Raises:
            ProviderError: Translated from raw OpenAI SDK exceptions.
        """
        import openai

        kwargs = self._build_kwargs(request)
        t0 = time.monotonic()
        try:
            raw = await self._client.responses.create(**kwargs)
        except openai.APIStatusError as exc:
            retry_after = _extract_retry_after(exc)
            raise error_from_status(
                exc.status_code,
                str(exc),
                provider="openai",
                retry_after=retry_after,
            ) from exc
        except openai.APITimeoutError as exc:
            raise RequestTimeoutError(
                str(exc), provider="openai", retryable=True
            ) from exc
        except openai.APIConnectionError as exc:
            raise NetworkError(str(exc)) from exc
        latency = (time.monotonic() - t0) * 1000
        return self._map_response(raw, latency)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a streaming request to the OpenAI Responses API.

        Args:
            request: Provider-agnostic request to send.

        Yields:
            StreamEvent instances as they arrive from the API.

        Raises:
            ProviderError: Translated from raw OpenAI SDK exceptions.
        """
        import openai

        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        t0 = time.monotonic()
        try:
            raw_stream = await self._client.responses.create(**kwargs)
        except openai.APIStatusError as exc:
            retry_after = _extract_retry_after(exc)
            raise error_from_status(
                exc.status_code,
                str(exc),
                provider="openai",
                retry_after=retry_after,
            ) from exc
        except openai.APITimeoutError as exc:
            raise RequestTimeoutError(
                str(exc), provider="openai", retryable=True
            ) from exc
        except openai.APIConnectionError as exc:
            raise NetworkError(str(exc)) from exc

        yield StreamEvent(type=StreamEventType.STREAM_START)

        pending_tools: dict[str, ToolCallContent] = {}

        async for event in raw_stream:
            event_type = event.type

            if event_type == "response.output_text.delta":
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text=event.delta,
                )

            elif event_type == "response.reasoning_summary_text.delta":
                yield StreamEvent(
                    type=StreamEventType.REASONING_DELTA,
                    text=event.delta,
                )

            elif event_type == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    tc = ToolCallContent(
                        tool_call_id=getattr(item, "call_id", "") or "",
                        tool_name=getattr(item, "name", "") or "",
                    )
                    pending_tools[item.id or event.item_id] = tc
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call=tc,
                    )

            elif event_type == "response.function_call_arguments.delta":
                item_id = event.item_id
                if item_id in pending_tools:
                    pending_tools[item_id].arguments_json += event.delta
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        text=event.delta,
                        tool_call=pending_tools[item_id],
                    )

            elif event_type == "response.function_call_arguments.done":
                item_id = event.item_id
                tc = pending_tools.pop(item_id, None)
                if tc:
                    tc.arguments_json = event.arguments
                    try:
                        tc.arguments = json.loads(event.arguments)
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("Failed to parse tool call arguments: %s", exc)
                    tc.tool_name = tc.tool_name or event.name
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        tool_call=tc,
                    )

            elif event_type == "response.completed":
                resp = event.response
                usage = None
                if resp and resp.usage:
                    reasoning_tokens = 0
                    if resp.usage.output_tokens_details:
                        reasoning_tokens = (
                            getattr(
                                resp.usage.output_tokens_details,
                                "reasoning_tokens",
                                0,
                            )
                            or 0
                        )
                    usage = TokenUsage(
                        input_tokens=resp.usage.input_tokens or 0,
                        output_tokens=resp.usage.output_tokens or 0,
                        reasoning_tokens=reasoning_tokens,
                    )

                has_tool_calls = any(
                    getattr(item, "type", None) == "function_call"
                    for item in (resp.output if resp else [])
                )
                finish = (
                    FinishReason("tool_calls", raw="tool_calls")
                    if has_tool_calls
                    else FinishReason("stop", raw="completed")
                )

                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=finish,
                    usage=usage,
                    metadata={"latency_ms": (time.monotonic() - t0) * 1000},
                )

            elif event_type == "response.failed":
                error_msg = ""
                if hasattr(event, "response") and event.response:
                    err = getattr(event.response, "error", None)
                    if err:
                        error_msg = getattr(err, "message", str(err))
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=error_msg or "Response generation failed",
                )

            elif event_type == "response.incomplete":
                yield StreamEvent(
                    type=StreamEventType.FINISH,
                    finish_reason=FinishReason("length", raw="incomplete"),
                    metadata={"latency_ms": (time.monotonic() - t0) * 1000},
                )
