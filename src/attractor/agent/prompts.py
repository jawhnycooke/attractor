"""System prompt construction and project documentation discovery.

Builds a layered system prompt with precedence (later overrides earlier):
1. Provider-specific base instructions
2. Environment context
3. Tool descriptions from active profile
4. Project docs (AGENTS.md + provider-specific files), 32 KB budget
5. User instruction overrides
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.profiles.base import ProviderProfile

# Max bytes of project documentation to include in the system prompt.
_PROJECT_DOC_BUDGET = 32 * 1024

# Provider-specific doc filenames to look for alongside AGENTS.md.
_PROVIDER_DOC_MAP: dict[str, list[str]] = {
    "anthropic": ["CLAUDE.md"],
    "openai": [".codex/instructions.md"],
    "google": ["GEMINI.md"],
}


def _git_root(working_dir: str) -> str | None:
    """Return the git repository root, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _git_branch(working_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "(not a git repo)"


def _git_status(working_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            return status if status else "clean"
    except Exception:
        pass
    return "(not a git repo)"


def discover_project_docs(working_dir: str, provider_name: str) -> str:
    """Walk from git root upward and load relevant project doc files.

    Always loads AGENTS.md if found, plus provider-specific files.
    Respects a 32 KB total budget.
    """
    root = _git_root(working_dir) or working_dir
    search_dirs = []

    # Walk from working_dir up to root
    current = Path(working_dir).resolve()
    root_path = Path(root).resolve()
    while True:
        search_dirs.append(current)
        if current == root_path or current.parent == current:
            break
        current = current.parent

    # Deduplicate while preserving order (closest first)
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for d in search_dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    # Files to look for
    filenames = ["AGENTS.md"] + _PROVIDER_DOC_MAP.get(provider_name, [])

    collected: list[str] = []
    total_bytes = 0

    for directory in unique_dirs:
        for fname in filenames:
            fpath = directory / fname
            if fpath.is_file():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    if total_bytes + len(content.encode()) > _PROJECT_DOC_BUDGET:
                        remaining = _PROJECT_DOC_BUDGET - total_bytes
                        if remaining > 0:
                            content = content[:remaining]
                            collected.append(
                                f"# {fname} (from {directory}, truncated)\n{content}"
                            )
                            total_bytes += remaining
                        break
                    collected.append(f"# {fname} (from {directory})\n{content}")
                    total_bytes += len(content.encode())
                except Exception:
                    continue
        if total_bytes >= _PROJECT_DOC_BUDGET:
            break

    return "\n\n".join(collected)


def build_system_prompt(
    profile: ProviderProfile,
    environment: ExecutionEnvironment,
    model_id: str = "",
    user_instructions: str = "",
) -> str:
    """Construct the full system prompt for an agent session.

    Layers:
    1. Provider base prompt with environment context
    2. Project documentation
    3. User instruction overrides
    """
    working_dir = environment.working_directory()
    platform = environment.platform()
    git_branch = _git_branch(working_dir)
    git_status = _git_status(working_dir)
    date = datetime.date.today().isoformat()

    project_docs = discover_project_docs(working_dir, profile.provider_name)

    prompt = profile.format_system_prompt(
        working_dir=working_dir,
        platform=platform,
        date=date,
        model_id=model_id,
        git_branch=git_branch,
        git_status=git_status,
        project_docs=project_docs,
        user_instructions=user_instructions,
    )

    return prompt
