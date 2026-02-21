"""Base provider profile — defines the interface all profiles implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from attractor.llm.models import ToolDefinition


class ProviderProfile(ABC):
    """Provider-specific configuration for tools and system prompts.

    Each supported LLM provider (Anthropic, OpenAI, Google) has a profile
    that determines which tools are available and how the system prompt is
    structured.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short provider identifier (e.g. 'anthropic', 'openai', 'google')."""

    @property
    @abstractmethod
    def tool_definitions(self) -> list[ToolDefinition]:
        """Tool definitions this profile exposes to the model."""

    @property
    @abstractmethod
    def system_prompt_template(self) -> str:
        """Template string for the base system prompt.

        May contain ``{placeholders}`` filled by ``format_system_prompt``.
        """

    @property
    @abstractmethod
    def context_window_size(self) -> int:
        """Maximum context window size in tokens for this profile."""

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
