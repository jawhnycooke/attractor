"""Tests for provider profiles: capability flags, provider_options, and protocol."""

from __future__ import annotations

from attractor.agent.profiles.anthropic_profile import AnthropicProfile
from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.profiles.gemini_profile import GeminiProfile
from attractor.agent.profiles.openai_profile import OpenAIProfile


class TestProviderProfileProtocol:
    """Verify the ProviderProfile protocol defines all required members."""

    def test_protocol_has_supports_reasoning(self) -> None:
        """Protocol must declare supports_reasoning property."""
        assert hasattr(ProviderProfile, "supports_reasoning")

    def test_protocol_has_supports_streaming(self) -> None:
        """Protocol must declare supports_streaming property."""
        assert hasattr(ProviderProfile, "supports_streaming")

    def test_protocol_has_supports_parallel_tool_calls(self) -> None:
        """Protocol must declare supports_parallel_tool_calls property."""
        assert hasattr(ProviderProfile, "supports_parallel_tool_calls")

    def test_protocol_has_provider_options(self) -> None:
        """Protocol must declare provider_options method."""
        assert hasattr(ProviderProfile, "provider_options")

    def test_protocol_has_context_window_size(self) -> None:
        """Protocol must declare context_window_size property."""
        assert hasattr(ProviderProfile, "context_window_size")

    def test_protocol_has_provider_name(self) -> None:
        """Protocol must declare provider_name property."""
        assert hasattr(ProviderProfile, "provider_name")

    def test_protocol_has_tool_definitions(self) -> None:
        """Protocol must declare tool_definitions property."""
        assert hasattr(ProviderProfile, "tool_definitions")

    def test_protocol_has_system_prompt_template(self) -> None:
        """Protocol must declare system_prompt_template property."""
        assert hasattr(ProviderProfile, "system_prompt_template")

    def test_anthropic_is_provider_profile(self) -> None:
        """AnthropicProfile must satisfy the ProviderProfile protocol."""
        assert isinstance(AnthropicProfile(), ProviderProfile)

    def test_openai_is_provider_profile(self) -> None:
        """OpenAIProfile must satisfy the ProviderProfile protocol."""
        assert isinstance(OpenAIProfile(), ProviderProfile)

    def test_gemini_is_provider_profile(self) -> None:
        """GeminiProfile must satisfy the ProviderProfile protocol."""
        assert isinstance(GeminiProfile(), ProviderProfile)


class TestAnthropicProfileCapabilities:
    """Verify AnthropicProfile capability flags match spec §3.5."""

    def setup_method(self) -> None:
        self.profile = AnthropicProfile()

    def test_supports_reasoning_is_true(self) -> None:
        """Anthropic supports extended thinking."""
        assert self.profile.supports_reasoning is True

    def test_supports_streaming_is_true(self) -> None:
        """Anthropic supports streaming responses."""
        assert self.profile.supports_streaming is True

    def test_supports_parallel_tool_calls_is_true(self) -> None:
        """Anthropic supports parallel tool call execution."""
        assert self.profile.supports_parallel_tool_calls is True

    def test_context_window_size(self) -> None:
        """Anthropic context window is 200K tokens."""
        assert self.profile.context_window_size == 200_000

    def test_provider_name(self) -> None:
        """Provider name is 'anthropic'."""
        assert self.profile.provider_name == "anthropic"


class TestAnthropicProfileProviderOptions:
    """Verify AnthropicProfile.provider_options() returns expected structure."""

    def setup_method(self) -> None:
        self.profile = AnthropicProfile()
        self.options = self.profile.provider_options()

    def test_returns_dict(self) -> None:
        """provider_options() must return a dict, not None."""
        assert isinstance(self.options, dict)

    def test_has_anthropic_key(self) -> None:
        """Options must contain an 'anthropic' top-level key."""
        assert "anthropic" in self.options

    def test_has_beta_headers(self) -> None:
        """Anthropic options must contain beta_headers list."""
        assert "beta_headers" in self.options["anthropic"]

    def test_beta_headers_is_list(self) -> None:
        """beta_headers must be a list."""
        assert isinstance(self.options["anthropic"]["beta_headers"], list)

    def test_extended_thinking_header_present(self) -> None:
        """Beta headers must include extended thinking header."""
        headers = self.options["anthropic"]["beta_headers"]
        assert "extended-thinking-2025-04-14" in headers

    def test_output_128k_header_present(self) -> None:
        """Beta headers must include 128K output header."""
        headers = self.options["anthropic"]["beta_headers"]
        assert "output-128k-2025-02-19" in headers


class TestOpenAIProfileCapabilities:
    """Verify OpenAIProfile capability flags match spec §3.4."""

    def setup_method(self) -> None:
        self.profile = OpenAIProfile()

    def test_supports_reasoning_is_true(self) -> None:
        """OpenAI reasoning models support reasoning."""
        assert self.profile.supports_reasoning is True

    def test_supports_streaming_is_true(self) -> None:
        """OpenAI supports streaming responses."""
        assert self.profile.supports_streaming is True

    def test_supports_parallel_tool_calls_is_true(self) -> None:
        """OpenAI supports parallel tool call execution."""
        assert self.profile.supports_parallel_tool_calls is True

    def test_context_window_size(self) -> None:
        """OpenAI context window is 128K tokens."""
        assert self.profile.context_window_size == 128_000

    def test_provider_name(self) -> None:
        """Provider name is 'openai'."""
        assert self.profile.provider_name == "openai"


class TestOpenAIProfileProviderOptions:
    """Verify OpenAIProfile.provider_options() returns expected structure."""

    def setup_method(self) -> None:
        self.profile = OpenAIProfile()
        self.options = self.profile.provider_options()

    def test_returns_dict(self) -> None:
        """provider_options() must return a dict, not None."""
        assert isinstance(self.options, dict)

    def test_has_openai_key(self) -> None:
        """Options must contain an 'openai' top-level key."""
        assert "openai" in self.options

    def test_has_reasoning_config(self) -> None:
        """OpenAI options must contain reasoning configuration."""
        assert "reasoning" in self.options["openai"]

    def test_reasoning_effort_value(self) -> None:
        """Reasoning effort must be set to 'medium'."""
        assert self.options["openai"]["reasoning"]["effort"] == "medium"


class TestGeminiProfileCapabilities:
    """Verify GeminiProfile capability flags match spec §3.6."""

    def setup_method(self) -> None:
        self.profile = GeminiProfile()

    def test_supports_reasoning_is_true(self) -> None:
        """Gemini supports thinking capabilities."""
        assert self.profile.supports_reasoning is True

    def test_supports_streaming_is_true(self) -> None:
        """Gemini supports streaming responses."""
        assert self.profile.supports_streaming is True

    def test_supports_parallel_tool_calls_is_true(self) -> None:
        """Gemini supports parallel tool call execution."""
        assert self.profile.supports_parallel_tool_calls is True

    def test_context_window_size(self) -> None:
        """Gemini context window is 1M tokens."""
        assert self.profile.context_window_size == 1_000_000

    def test_provider_name(self) -> None:
        """Provider name is 'google'."""
        assert self.profile.provider_name == "google"


class TestGeminiProfileProviderOptions:
    """Verify GeminiProfile.provider_options() returns expected structure."""

    def setup_method(self) -> None:
        self.profile = GeminiProfile()
        self.options = self.profile.provider_options()

    def test_returns_dict(self) -> None:
        """provider_options() must return a dict, not None."""
        assert isinstance(self.options, dict)

    def test_has_gemini_key(self) -> None:
        """Options must contain a 'gemini' top-level key."""
        assert "gemini" in self.options

    def test_has_safety_settings(self) -> None:
        """Gemini options must contain safety_settings."""
        assert "safety_settings" in self.options["gemini"]

    def test_safety_settings_is_list(self) -> None:
        """safety_settings must be a list."""
        assert isinstance(self.options["gemini"]["safety_settings"], list)
