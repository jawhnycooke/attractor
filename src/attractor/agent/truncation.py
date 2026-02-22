"""Two-stage tool output truncation pipeline.

Stage 1 — Character-based (mandatory): Either head/tail split preserving
beginning and end of the output while omitting the middle, or tail-only
mode which discards the beginning and keeps the end. Each tool has a
configured mode (``head_tail`` or ``tail``).

Stage 2 — Line-based (optional, applied after stage 1): head/tail split
operating on lines.

Each tool has its own configurable limits and truncation mode.
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
    char_limits: dict[str, int] = field(
        default_factory=lambda: {
            "read_file": 50_000,
            "shell": 30_000,
            "grep": 20_000,
            "glob": 20_000,
            "spawn_agent": 20_000,
            "edit_file": 10_000,
            "apply_patch": 10_000,
            "write_file": 1_000,
        }
    )

    # Truncation mode per tool: "head_tail" or "tail"
    truncation_modes: dict[str, str] = field(
        default_factory=lambda: {
            "read_file": "head_tail",
            "shell": "head_tail",
            "grep": "tail",
            "glob": "tail",
            "edit_file": "tail",
            "apply_patch": "tail",
            "write_file": "tail",
            "spawn_agent": "head_tail",
        }
    )

    # Stage 2 — line limits (0 means skip this stage)
    line_limits: dict[str, int] = field(
        default_factory=lambda: {
            "shell": 256,
            "grep": 200,
            "glob": 500,
        }
    )


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
        f"\n\n[WARNING: Tool output was truncated. "
        f"{removed} characters were removed from the middle. "
        f"The full output is available in the event stream. "
        f"If you need to see specific parts, re-run the tool "
        f"with more targeted parameters.]\n\n"
    )
    return head + marker + tail


def _truncate_tail(text: str, max_chars: int) -> str:
    """Truncate *text* by discarding the beginning, keeping only the tail."""
    if len(text) <= max_chars:
        return text

    removed = len(text) - max_chars
    marker = (
        f"[WARNING: Tool output was truncated. First "
        f"{removed} characters were removed. "
        f"The full output is available in the event stream.]\n\n"
    )
    return marker + text[-max_chars:]


def _truncate_lines(text: str, max_lines: int) -> str:
    """Truncate *text* by line count, keeping head and tail lines."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text

    keep = max_lines // 2
    removed = len(lines) - max_lines
    head = lines[:keep]
    tail = lines[-keep:]
    marker = f"\n[... {removed} lines omitted ...]\n"
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
        mode = cfg.truncation_modes.get(tool_name, "head_tail")
        if mode == "tail":
            result = _truncate_tail(result, char_limit)
        else:
            result = _truncate_chars(result, char_limit)

    # Stage 2 — line truncation
    line_limit = cfg.line_limits.get(tool_name, 0)
    if line_limit > 0:
        result = _truncate_lines(result, line_limit)

    return result, full_original
