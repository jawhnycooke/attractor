"""Google (gemini-cli) provider profile.

Uses edit_file with Gemini conventions. Includes list_dir and
optional web tools.
"""

from __future__ import annotations

from attractor.agent.tools.core_tools import (
    EDIT_FILE_DEF,
    GLOB_DEF,
    GREP_DEF,
    LIST_DIR_DEF,
    READ_FILE_DEF,
    SHELL_DEF,
    WRITE_FILE_DEF,
)
from attractor.agent.tools.subagent import SPAWN_AGENT_DEF
from attractor.llm.models import ToolDefinition

_SYSTEM_PROMPT = """\
You are an AI coding assistant. You help users write, debug, and understand \
code by interacting with the filesystem and running commands.

# Environment
- Working directory: {working_dir}
- Platform: {platform}
- Date: {date}
- Model: {model_id}
- Git branch: {git_branch}
- Git status: {git_status}

# Guidelines
- Read files before editing. Use list_dir to understand project structure.
- Use edit_file for modifying existing files with search-and-replace.
- Use write_file for creating new files.
- Run shell commands to build, test, and verify changes.
- Use grep to search file contents and glob to find files by pattern.
- Be thorough but concise in explanations.

{project_docs}
{user_instructions}
"""


class GeminiProfile:
    """Profile aligned with gemini-cli conventions.

    Provides Google-specific tool definitions and system prompt
    template for Gemini models. Includes list_dir for directory
    exploration.
    """

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "google"

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        """Return Gemini tool definitions including list_dir."""
        return [
            EDIT_FILE_DEF,
            READ_FILE_DEF,
            WRITE_FILE_DEF,
            SHELL_DEF,
            GREP_DEF,
            GLOB_DEF,
            LIST_DIR_DEF,
            SPAWN_AGENT_DEF,
        ]

    @property
    def system_prompt_template(self) -> str:
        """Return the gemini-cli system prompt template."""
        return _SYSTEM_PROMPT

    @property
    def context_window_size(self) -> int:
        """Return the 1M context window size for Gemini models."""
        return 1_000_000

    @property
    def supports_reasoning(self) -> bool:
        """Return True — Gemini supports thinking capabilities."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Return True — Gemini supports streaming responses."""
        return True

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Return True — Gemini supports parallel tool call execution."""
        return True

    @property
    def default_timeout_ms(self) -> int:
        """Return 10s default shell timeout."""
        return 10_000

    def provider_options(self) -> dict | None:
        """Return Gemini-specific options including safety settings.

        Returns:
            Dict with safety settings configuration for Gemini models.
        """
        return {
            "gemini": {
                "safety_settings": [],
            },
        }

    def get_tools(self) -> list[ToolDefinition]:
        """Return the tool definitions for this profile."""
        return list(self.tool_definitions)

    def format_system_prompt(self, **kwargs: str) -> str:
        """Render the system prompt template with the given variables."""
        template = self.system_prompt_template
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
