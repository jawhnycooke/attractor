# CLAUDE.md — LLM Client Subsystem

The LLM subsystem provides a provider-agnostic interface for Anthropic (Claude), OpenAI (GPT), and Google (Gemini). It handles model routing, middleware pipelines, retry logic, streaming, and tool-use loops.

## Module Map

| Module | Purpose |
|---|---|
| `models.py` | Provider-neutral data model: `Message`, `Request`, `Response`, `ToolDefinition`, `StreamEvent`, content types, enums. |
| `client.py` | `LLMClient` — routes by model name, applies middleware, retries, concurrent tool execution. |
| `middleware.py` | `Middleware` protocol for request/response transformation. |
| `streaming.py` | `StreamCollector` — aggregates `StreamEvent`s into a final `Response`. |
| `adapters/base.py` | `ProviderAdapter` protocol: `detect_model`, `complete`, `stream`. |
| `adapters/anthropic_adapter.py` | Anthropic Messages API. Enforces user/assistant alternation, maps reasoning effort to thinking budget. |
| `catalog.py` | Model catalog: maps model IDs to provider metadata, capabilities, and context window sizes. |
| `adapters/openai_adapter.py` | OpenAI Responses API (`client.responses.create()`). |
| `adapters/gemini_adapter.py` | Google Gemini API. |

## Key Patterns

- **Adapter lazy-loading**: `LLMClient._default_adapters()` tries to import each adapter; silently skips on `ImportError`. No hard dependency on any provider SDK.
- **Model detection**: Each adapter's `detect_model(model_str)` checks if the model string belongs to that provider (e.g., `claude-*` → Anthropic, `gpt-*` → OpenAI).
- **Middleware pipeline**: `before_request` applied in order, `after_response` in reverse order. Used for logging, caching, content filtering.
- **Retry policy**: `RetryPolicy` with exponential backoff. Applied at the **client level** (`LLMClient._complete_with_retry`), transparent to callers.
- **Concurrent tool execution**: `generate()` dispatches multiple tool calls per round via `asyncio.gather`, feeds results back, loops up to `max_tool_rounds`.
- **Streaming**: `stream()` returns `AsyncIterator[StreamEvent]`. `StreamCollector` can aggregate into a `Response`. `stream_generate()` is single-round only (no tool loop).
- **Structured output**: `generate_object()` uses `response_format` with JSON schema for constrained generation.

## Content Type System

Messages contain a list of `ContentPart` items, each tagged with a `ContentKind`:

- `TEXT` — Plain text
- `TOOL_CALL` — Tool invocation (name, arguments, ID)
- `TOOL_RESULT` — Tool output (result string, optional error flag)
- `THINKING` / `REDACTED_THINKING` — Model reasoning (Anthropic-specific)
- `IMAGE`, `AUDIO`, `DOCUMENT` — Multimodal content

Helper methods: `Message.user(text)`, `Message.assistant(text)`, `Message.tool_result(id, result)`, `message.text()`, `message.tool_calls()`.

## Adding a New Provider Adapter

1. Create `adapters/new_provider_adapter.py`.
2. Implement the `ProviderAdapter` protocol: `provider_name()`, `detect_model()`, `complete()`, `stream()`.
3. Map provider-specific response formats to `Response` / `StreamEvent`.
4. Add the import to `LLMClient._default_adapters()` with a try/except guard.
5. Add a corresponding `ProviderProfile` in `agent/profiles/` if the provider needs a custom tool set or system prompt.

## Anthropic-Specific Quirks

- The API requires strict user/assistant message alternation. The adapter inserts synthetic placeholder messages when consecutive same-role messages are detected.
- `reasoning_effort` maps to extended thinking budget parameters.
- Tool use is returned as `tool_use` content blocks, not function calls.
