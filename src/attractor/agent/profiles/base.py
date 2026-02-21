"""Base provider profile — defines the interface all profiles implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from attractor.llm.models import ToolDefinition


@runtime_checkable
class ProviderProfile(Protocol):
    """Provider-specific configuration for tools and system prompts.

    Each supported LLM provider (Anthropic, OpenAI, Google) has a profile
    that determines which tools are available and how the system prompt is
    structured.

    Concrete implementations must provide all four properties. The
    ``get_tools`` and ``format_system_prompt`` methods have default
    implementations that can be overridden if needed.
    """

    @property
    def provider_name(self) -> str:
        """Short provider identifier (e.g. 'anthropic', 'openai', 'google')."""
        ...

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        """Tool definitions this profile exposes to the model."""
        ...

    @property
    def system_prompt_template(self) -> str:
        """Template string for the base system prompt.

        May contain ``{placeholders}`` filled by ``format_system_prompt``.
        """
        ...

    @property
    def context_window_size(self) -> int:
        """Maximum context window size in tokens for this profile."""
        ...

    def get_tools(self) -> list[ToolDefinition]:
        """Return the tool definitions for this profile."""
        return list(self.tool_definitions)

    def format_system_prompt(self, **kwargs: str) -> str:
        """Render the system prompt template with the given variables.

        Common variables: working_dir, platform, date, model_id,
        git_branch, git_status.
        """
        template = self.system_prompt_template
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
