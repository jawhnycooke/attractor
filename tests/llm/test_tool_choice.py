"""Tests for ToolChoice translation across all 3 LLM adapters.

Verifies that the unified ToolChoice dataclass is correctly mapped to each
provider's native tool_choice format per unified-llm-spec §4.3.
"""

from __future__ import annotations

from attractor.llm.adapters.anthropic_adapter import AnthropicAdapter
from attractor.llm.adapters.openai_adapter import OpenAIAdapter
from attractor.llm.adapters.gemini_adapter import GeminiAdapter
from attractor.llm.models import ToolChoice


# ---------------------------------------------------------------------------
# Anthropic ToolChoice mapping
# ---------------------------------------------------------------------------


class TestAnthropicToolChoice:
    def test_auto_maps_to_type_auto(self) -> None:
        result = AnthropicAdapter._map_tool_choice(ToolChoice(mode="auto"))
        assert result == {"type": "auto"}

    def test_none_returns_none_for_tools_removal(self) -> None:
        """Anthropic: mode='none' returns None; caller pops tools from kwargs."""
        result = AnthropicAdapter._map_tool_choice(ToolChoice(mode="none"))
        assert result is None

    def test_required_maps_to_type_any(self) -> None:
        result = AnthropicAdapter._map_tool_choice(ToolChoice(mode="required"))
        assert result == {"type": "any"}

    def test_named_maps_to_type_tool_with_name(self) -> None:
        result = AnthropicAdapter._map_tool_choice(
            ToolChoice(mode="named", tool_name="read_file")
        )
        assert result == {"type": "tool", "name": "read_file"}

    def test_none_input_returns_none(self) -> None:
        result = AnthropicAdapter._map_tool_choice(None)
        assert result is None


class TestAnthropicNormalizeToolChoice:
    """Test the _normalize_tool_choice helper used before mapping."""

    def test_string_normalized_to_tool_choice(self) -> None:
        result = AnthropicAdapter._normalize_tool_choice("required")
        assert isinstance(result, ToolChoice)
        assert result.mode == "required"

    def test_dict_normalized_to_tool_choice(self) -> None:
        result = AnthropicAdapter._normalize_tool_choice(
            {"mode": "named", "tool_name": "foo"}
        )
        assert isinstance(result, ToolChoice)
        assert result.mode == "named"
        assert result.tool_name == "foo"

    def test_tool_choice_passed_through(self) -> None:
        tc = ToolChoice(mode="auto")
        result = AnthropicAdapter._normalize_tool_choice(tc)
        assert result is tc

    def test_none_returns_none(self) -> None:
        result = AnthropicAdapter._normalize_tool_choice(None)
        assert result is None


# ---------------------------------------------------------------------------
# OpenAI ToolChoice mapping
# ---------------------------------------------------------------------------


class TestOpenAIToolChoice:
    def test_auto_maps_to_string_auto(self) -> None:
        result = OpenAIAdapter._map_tool_choice(ToolChoice(mode="auto"))
        assert result == "auto"

    def test_none_returns_sentinel_for_tools_removal(self) -> None:
        """OpenAI: mode='none' returns '__none__' sentinel; caller pops tools."""
        result = OpenAIAdapter._map_tool_choice(ToolChoice(mode="none"))
        assert result == "__none__"

    def test_required_maps_to_string_required(self) -> None:
        result = OpenAIAdapter._map_tool_choice(ToolChoice(mode="required"))
        assert result == "required"

    def test_named_maps_to_function_object(self) -> None:
        result = OpenAIAdapter._map_tool_choice(
            ToolChoice(mode="named", tool_name="search")
        )
        assert result == {
            "type": "function",
            "function": {"name": "search"},
        }

    def test_none_input_returns_none(self) -> None:
        result = OpenAIAdapter._map_tool_choice(None)
        assert result is None


class TestOpenAIToolChoiceFromRawTypes:
    """Test that string and dict inputs are also accepted."""

    def test_string_auto(self) -> None:
        result = OpenAIAdapter._map_tool_choice("auto")
        assert result == "auto"

    def test_dict_named(self) -> None:
        result = OpenAIAdapter._map_tool_choice(
            {"mode": "named", "tool_name": "write"}
        )
        assert result == {
            "type": "function",
            "function": {"name": "write"},
        }


# ---------------------------------------------------------------------------
# Gemini ToolChoice mapping
# ---------------------------------------------------------------------------


class TestGeminiToolChoice:
    def test_auto_maps_to_mode_auto(self) -> None:
        result = GeminiAdapter._map_tool_choice(ToolChoice(mode="auto"))
        assert result == {"mode": "AUTO"}

    def test_none_returns_sentinel_for_tools_removal(self) -> None:
        """Gemini: mode='none' returns '__none__' sentinel; caller pops tools."""
        result = GeminiAdapter._map_tool_choice(ToolChoice(mode="none"))
        assert result == "__none__"

    def test_required_maps_to_mode_any(self) -> None:
        result = GeminiAdapter._map_tool_choice(ToolChoice(mode="required"))
        assert result == {"mode": "ANY"}

    def test_named_maps_to_mode_any_with_allowed_names(self) -> None:
        result = GeminiAdapter._map_tool_choice(
            ToolChoice(mode="named", tool_name="execute_code")
        )
        assert result == {
            "mode": "ANY",
            "allowed_function_names": ["execute_code"],
        }

    def test_none_input_returns_none(self) -> None:
        result = GeminiAdapter._map_tool_choice(None)
        assert result is None


class TestGeminiToolChoiceFromRawTypes:
    """Test that string and dict inputs are also accepted."""

    def test_string_required(self) -> None:
        result = GeminiAdapter._map_tool_choice("required")
        assert result == {"mode": "ANY"}

    def test_dict_auto(self) -> None:
        result = GeminiAdapter._map_tool_choice({"mode": "auto"})
        assert result == {"mode": "AUTO"}
