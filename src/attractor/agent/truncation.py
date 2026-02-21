"""Two-stage tool output truncation pipeline.

Stage 1 — Character-based (mandatory): head/tail split preserving beginning
and end of the output while omitting the middle.

Stage 2 — Line-based (optional, applied after stage 1): same head/tail
split pattern but operating on lines.

Each tool has its own configurable limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TruncationConfig:
    """Per-tool truncation limits.

    Set a limit to 0 to disable that stage for a given tool.
    """

    # Stage 1 — character limits
    char_limits: dict[str, int] = field(default_factory=lambda: {
        "read_file": 50_000,
        "shell": 30_000,
        "grep": 20_000,
        "glob": 20_000,
        "edit_file": 10_000,
        "apply_patch": 10_000,
        "write_file": 1_000,
    })

    # Stage 2 — line limits (0 means skip this stage)
    line_limits: dict[str, int] = field(default_factory=lambda: {
        "shell": 256,
        "grep": 200,
        "glob": 500,
    })


_DEFAULT_CONFIG = TruncationConfig()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate_chars(text: str, max_chars: int) -> str:
    """Truncate *text* by characters, keeping head and tail."""
    if len(text) <= max_chars:
        return text

    keep = max_chars // 2
    removed = len(text) - max_chars
    head = text[:keep]
    tail = text[-keep:]
    marker = (
        f"\n[WARNING: Tool output was truncated. "
        f"{removed} characters removed from middle of output]\n"
    )
    return head + marker + tail


def _truncate_lines(text: str, max_lines: int) -> str:
    """Truncate *text* by line count, keeping head and tail lines."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text

    keep = max_lines // 2
    removed = len(lines) - max_lines
    head = lines[:keep]
    tail = lines[-keep:]
    marker = (
        f"\n[WARNING: Tool output was truncated. "
        f"{removed} lines removed from middle of output]\n"
    )
    return "".join(head) + marker + "".join(tail)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def truncate_output(
    tool_name: str,
    output: str,
    config: TruncationConfig | None = None,
) -> tuple[str, str]:
    """Apply the two-stage truncation pipeline.

    Args:
        tool_name: Name of the tool whose output is being truncated.
        output: Raw tool output string.
        config: Optional overrides; uses sensible defaults otherwise.

    Returns:
        A ``(truncated, full_original)`` tuple. *truncated* is suitable for
        sending to the LLM; *full_original* is the unmodified output.
    """
    cfg = config or _DEFAULT_CONFIG
    full_original = output
    result = output

    # Stage 1 — character truncation
    char_limit = cfg.char_limits.get(tool_name, 0)
    if char_limit > 0:
        result = _truncate_chars(result, char_limit)

    # Stage 2 — line truncation
    line_limit = cfg.line_limits.get(tool_name, 0)
    if line_limit > 0:
        result = _truncate_lines(result, line_limit)

    return result, full_original
