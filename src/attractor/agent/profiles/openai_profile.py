"""OpenAI (codex-rs) provider profile.

Uses apply_patch instead of edit_file. System prompt follows
codex-rs conventions.
"""

from __future__ import annotations

from attractor.agent.profiles.base import ProviderProfile
from attractor.agent.tools.apply_patch import APPLY_PATCH_DEF
from attractor.agent.tools.core_tools import (
    GLOB_DEF,
    GREP_DEF,
    READ_FILE_DEF,
    SHELL_DEF,
    WRITE_FILE_DEF,
)
from attractor.agent.tools.subagent import SPAWN_AGENT_DEF
from attractor.llm.models import ToolDefinition

_SYSTEM_PROMPT = """\
You are an autonomous coding agent operating in a sandboxed environment. \
You solve programming tasks by reading files, applying patches, running \
commands, and searching code.

# Environment
- Working directory: {working_dir}
- Platform: {platform}
- Date: {date}
- Model: {model_id}
- Git branch: {git_branch}
- Git status: {git_status}

# Guidelines
- Always read a file before modifying it.
- Use apply_patch for all code edits using v4a unified diff format.
- Use write_file only for creating entirely new files.
- Run tests after making changes to verify correctness.
- Prefer precise, minimal changes.
- Use grep and glob to navigate the codebase efficiently.

{project_docs}
{user_instructions}
"""


class OpenAIProfile(ProviderProfile):
    """Profile aligned with codex-rs conventions."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [
            APPLY_PATCH_DEF,
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
        return 128_000
