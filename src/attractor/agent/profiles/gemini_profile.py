"""Google (gemini-cli) provider profile.

Uses edit_file with Gemini conventions. Includes list_dir and
optional web tools.
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

_LIST_DIR_DEF = ToolDefinition(
    name="list_dir",
    description="List directory contents with file sizes.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list.",
            },
            "depth": {
                "type": "integer",
                "description": "How many levels deep to list (default 1).",
            },
        },
        "required": ["path"],
    },
)

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


class GeminiProfile(ProviderProfile):
    """Profile aligned with gemini-cli conventions."""

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [
            EDIT_FILE_DEF,
            READ_FILE_DEF,
            WRITE_FILE_DEF,
            SHELL_DEF,
            GREP_DEF,
            GLOB_DEF,
            _LIST_DIR_DEF,
            SPAWN_AGENT_DEF,
        ]

    @property
    def system_prompt_template(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def context_window_size(self) -> int:
        return 1_000_000
