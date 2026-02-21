"""Model catalog for the Unified LLM Client.

Provides a registry of known models with capabilities, costs, and
lookup functions. The catalog is advisory -- unknown model strings
are still passed through to the provider.
"""

from __future__ import annotations

from attractor.llm.models import ModelInfo

# Current model catalog (February 2026)
MODELS: list[ModelInfo] = [
    # Anthropic
    ModelInfo(
        model_id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200_000,
        max_output_tokens=32_768,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["opus", "claude-opus"],
    ),
    ModelInfo(
        model_id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["sonnet", "claude-sonnet"],
    ),
    # OpenAI
    ModelInfo(
        model_id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=1_047_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt5", "gpt-5"],
    ),
    ModelInfo(
        model_id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt5-mini"],
    ),
    ModelInfo(
        model_id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=1_047_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["codex"],
    ),
    # Gemini
    ModelInfo(
        model_id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro (Preview)",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-pro", "gemini-3-pro"],
    ),
    ModelInfo(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash (Preview)",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-flash", "gemini-3-flash"],
    ),
]

# Build lookup index
_MODEL_INDEX: dict[str, ModelInfo] = {}
for _m in MODELS:
    _MODEL_INDEX[_m.model_id] = _m
    for _alias in _m.aliases:
        _MODEL_INDEX[_alias] = _m


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up a model by ID or alias. Returns None if unknown."""
    return _MODEL_INDEX.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List all known models, optionally filtered by provider."""
    if provider is None:
        return list(MODELS)
    return [m for m in MODELS if m.provider == provider]


def get_latest_model(
    provider: str, capability: str | None = None
) -> ModelInfo | None:
    """Get the first (latest/best) model for a provider.

    Optionally filter by capability: "reasoning", "vision", "tools".
    Returns None if no matching model found.
    """
    for m in MODELS:
        if m.provider != provider:
            continue
        if capability == "reasoning" and not m.supports_reasoning:
            continue
        if capability == "vision" and not m.supports_vision:
            continue
        if capability == "tools" and not m.supports_tools:
            continue
        return m
    return None
