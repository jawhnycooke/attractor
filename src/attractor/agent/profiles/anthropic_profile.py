"""Anthropic (Claude Code) provider profile.

Uses edit_file with old_string/new_string for code modifications.
System prompt follows Claude Code conventions.
"""

from __future__ import annotations

from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.core_tools import (
    EDIT_FILE_DEF,
    GLOB_DEF,
    GREP_DEF,
    READ_FILE_DEF,
    SHELL_DEF,
    WRITE_FILE_DEF,
)
from attractor.agent.tools.subagent import SPAWN_AGENT_DEF
from attractor.llm.models import ToolDefinition

_SYSTEM_PROMPT = """\
You are an autonomous coding agent. You help users with software engineering \
tasks by reading, writing, and editing code files, running shell commands, \
and searching the codebase.

# Environment
- Working directory: {working_dir}
- Platform: {platform}
- Date: {date}
- Model: {model_id}
- Git branch: {git_branch}
- Git status: {git_status}

# Guidelines
- Read files before editing them. Never guess at file contents.
- Use edit_file for precise search-and-replace edits. The old_string must \
be unique in the file.
- Use write_file only for creating new files or complete rewrites.
- Prefer targeted shell commands over broad operations.
- When searching code, use grep for content and glob for file paths.
- Keep changes minimal and focused on the task at hand.
- Test changes when possible by running relevant commands.

{project_docs}
{user_instructions}
"""


class AnthropicProfile(ProviderProfile):
    """Profile aligned with Claude Code conventions."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [
            EDIT_FILE_DEF,
            READ_FILE_DEF,
            WRITE_FILE_DEF,
            SHELL_DEF,
            GREP_DEF,
            GLOB_DEF,
            SPAWN_AGENT_DEF,
        ]

    @property
    def system_prompt_template(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def context_window_size(self) -> int:
        return 200_000
