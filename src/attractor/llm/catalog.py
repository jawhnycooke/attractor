"""Model catalog for the Unified LLM Client.

Provides a registry of known models with capabilities, costs, and
lookup functions. The catalog is advisory -- unknown model strings
are still passed through to the provider.
"""

from __future__ import annotations

from attractor.llm.models import ModelInfo

# Current model catalog (February 2026)
MODELS: list[ModelInfo] = [
    # ── Anthropic ──────────────────────────────────────────────────────
    ModelInfo(
        model_id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200_000,
        max_output_tokens=128_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=5.0,
        output_cost_per_million=25.0,
        aliases=["opus", "claude-opus"],
    ),
    ModelInfo(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        display_name="Claude Sonnet 4.6",
        context_window=200_000,
        max_output_tokens=64_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        aliases=["sonnet", "claude-sonnet"],
    ),
    ModelInfo(
        model_id="claude-opus-4-5",
        provider="anthropic",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        max_output_tokens=64_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=5.0,
        output_cost_per_million=25.0,
        aliases=["claude-opus-4-5"],
    ),
    ModelInfo(
        model_id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        max_output_tokens=64_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        aliases=["claude-sonnet-4-5"],
    ),
    ModelInfo(
        model_id="claude-haiku-4-5",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        max_output_tokens=64_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=1.0,
        output_cost_per_million=5.0,
        aliases=["haiku", "claude-haiku"],
    ),
    # ── OpenAI ─────────────────────────────────────────────────────────
    ModelInfo(
        model_id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        input_cost_per_million=1.75,
        output_cost_per_million=14.0,
        aliases=["gpt5", "gpt-5"],
    ),
    ModelInfo(
        model_id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt5-mini"],
    ),
    ModelInfo(
        model_id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt-codex"],
    ),
    ModelInfo(
        model_id="codex-mini-latest",
        provider="openai",
        display_name="Codex Mini (Latest)",
        context_window=128_000,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=False,
        supports_reasoning=True,
        input_cost_per_million=1.5,
        output_cost_per_million=6.0,
        aliases=["codex-mini"],
    ),
    ModelInfo(
        model_id="gpt-4.1",
        provider="openai",
        display_name="GPT-4.1",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=False,
        aliases=["gpt4.1"],
    ),
    ModelInfo(
        model_id="o3",
        provider="openai",
        display_name="OpenAI o3",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["openai-o3"],
    ),
    ModelInfo(
        model_id="o4-mini",
        provider="openai",
        display_name="OpenAI o4-mini",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["openai-o4-mini"],
    ),
    # ── Gemini ─────────────────────────────────────────────────────────
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
    ModelInfo(
        model_id="gemini-2.5-pro",
        provider="gemini",
        display_name="Gemini 2.5 Pro",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-2.5-pro"],
    ),
    ModelInfo(
        model_id="gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-2.5-flash"],
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
