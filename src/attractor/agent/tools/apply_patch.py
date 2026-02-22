"""apply_patch tool — v4a format parser for the OpenAI codex profile.

Parses the v4a patch format into discrete operations (AddFile, DeleteFile,
UpdateFile with optional Move) and applies them to the filesystem via the
ExecutionEnvironment.

v4a grammar (from spec Appendix A):

    patch       = "*** Begin Patch\n" operations "*** End Patch\n"
    operations  = (add_file | delete_file | update_file)*
    add_file    = "*** Add File: " path "\n" added_lines
    delete_file = "*** Delete File: " path "\n"
    update_file = "*** Update File: " path "\n" [move_line] hunks
    move_line   = "*** Move to: " new_path "\n"
    added_lines = ("+" line "\n")*
    hunks       = hunk+
    hunk        = "@@ " [context_hint] "\n" hunk_lines
    hunk_lines  = (context_line | delete_line | add_line)+
    context_line = " " line "\n"
    delete_line  = "-" line "\n"
    add_line     = "+" line "\n"
    eof_marker   = "*** End of File\n"
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import ToolDefinition

# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


class OpKind(Enum):
    ADD_FILE = auto()
    DELETE_FILE = auto()
    UPDATE_FILE = auto()
    MOVE_FILE = auto()


@dataclass
class Hunk:
    """A single hunk within an UpdateFile operation."""

    context_hint: str | None = None
    context_before: list[str] = field(default_factory=list)
    removals: list[str] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class PatchOp:
    """A single file-level operation parsed from patch text."""

    kind: OpKind
    path: str
    new_path: str | None = None  # for MOVE_FILE
    content: str | None = None  # for ADD_FILE
    hunks: list[Hunk] = field(default_factory=list)  # for UPDATE_FILE


# ---------------------------------------------------------------------------
# v4a Parser
# ---------------------------------------------------------------------------

_BEGIN_PATCH = "*** Begin Patch"
_END_PATCH = "*** End Patch"
_ADD_FILE = "*** Add File: "
_DELETE_FILE = "*** Delete File: "
_UPDATE_FILE = "*** Update File: "
_MOVE_TO = "*** Move to: "
_HUNK_MARKER = "@@ "
_EOF_MARKER = "*** End of File"


def parse_patch(text: str) -> list[PatchOp]:
    """Parse v4a-format patch text into a list of PatchOp."""
    ops: list[PatchOp] = []
    lines = text.splitlines()
    i = 0

    # Advance past optional leading whitespace / the Begin Patch marker
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == _BEGIN_PATCH:
            i += 1
            break
        if stripped.startswith("*** ") or stripped.startswith("@@ "):
            # No explicit Begin Patch but looks like v4a content
            break
        i += 1

    while i < len(lines):
        line = lines[i].rstrip()

        if line.strip() == _END_PATCH:
            break

        # --- Add File ---
        if line.startswith(_ADD_FILE):
            path = line[len(_ADD_FILE) :].strip()
            i += 1
            content_lines: list[str] = []
            while i < len(lines):
                raw = lines[i].rstrip()
                if raw.startswith("*** ") or raw.strip() == _END_PATCH:
                    break
                if raw.startswith("+"):
                    content_lines.append(raw[1:])
                i += 1
            ops.append(
                PatchOp(
                    kind=OpKind.ADD_FILE,
                    path=path,
                    content="\n".join(content_lines)
                    + ("\n" if content_lines else ""),
                )
            )
            continue

        # --- Delete File ---
        if line.startswith(_DELETE_FILE):
            path = line[len(_DELETE_FILE) :].strip()
            i += 1
            ops.append(PatchOp(kind=OpKind.DELETE_FILE, path=path))
            continue

        # --- Update File (possibly with Move) ---
        if line.startswith(_UPDATE_FILE):
            path = line[len(_UPDATE_FILE) :].strip()
            i += 1
            new_path: str | None = None

            # Check for optional Move line
            if i < len(lines) and lines[i].rstrip().startswith(_MOVE_TO):
                new_path = lines[i].rstrip()[len(_MOVE_TO) :].strip()
                i += 1

            # Parse hunks
            hunks: list[Hunk] = []
            while i < len(lines):
                raw = lines[i].rstrip()
                if raw.startswith("*** ") and not raw.startswith(_EOF_MARKER):
                    break
                if raw.strip() == _END_PATCH:
                    break
                if raw.startswith(_HUNK_MARKER):
                    hint_text = raw[len(_HUNK_MARKER) :].strip() or None
                    i += 1
                    hunk = Hunk(context_hint=hint_text)
                    in_changes = False
                    while i < len(lines):
                        hl = lines[i].rstrip()
                        if (
                            hl.startswith(_HUNK_MARKER)
                            or (hl.startswith("*** ") and not hl.startswith(_EOF_MARKER))
                            or hl.strip() == _END_PATCH
                        ):
                            break
                        if hl.startswith(_EOF_MARKER):
                            i += 1
                            break
                        if hl.startswith("-"):
                            hunk.removals.append(hl[1:])
                            in_changes = True
                        elif hl.startswith("+"):
                            hunk.additions.append(hl[1:])
                            in_changes = True
                        elif hl.startswith(" ") or hl == "":
                            # Space-prefixed = context line; empty = empty context
                            ctx_line = hl[1:] if hl.startswith(" ") else ""
                            if in_changes:
                                hunk.context_after.append(ctx_line)
                            else:
                                hunk.context_before.append(ctx_line)
                        else:
                            # Unknown prefix — skip
                            i += 1
                            continue
                        i += 1
                    hunks.append(hunk)
                else:
                    i += 1

            if new_path:
                ops.append(
                    PatchOp(
                        kind=OpKind.MOVE_FILE,
                        path=path,
                        new_path=new_path,
                        hunks=hunks,
                    )
                )
            else:
                ops.append(
                    PatchOp(
                        kind=OpKind.UPDATE_FILE,
                        path=path,
                        hunks=hunks,
                    )
                )
            continue

        i += 1

    return ops


# ---------------------------------------------------------------------------
# Applicator
# ---------------------------------------------------------------------------


def _normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return " ".join(s.split())


def _find_hunk_location(file_lines: list[str], hunk: Hunk) -> int | None:
    """Find where a hunk's context lines match in the file.

    Uses context_before + removals as the search pattern. If exact match
    fails, tries fuzzy matching via whitespace normalization.

    Returns the index of the first line of the match, or None if not found.
    """
    search_lines = hunk.context_before + hunk.removals
    if not search_lines:
        return 0

    # --- Exact match ---
    for i in range(len(file_lines)):
        if file_lines[i : i + len(search_lines)] == search_lines:
            return i

    # --- Fuzzy match: whitespace normalization ---
    normalized_search = [_normalize_whitespace(s) for s in search_lines]
    for i in range(len(file_lines)):
        candidate = [_normalize_whitespace(fl) for fl in file_lines[i : i + len(search_lines)]]
        if candidate == normalized_search:
            return i

    # --- Context hint match: use hint to narrow search area ---
    if hunk.context_hint:
        hint_norm = _normalize_whitespace(hunk.context_hint)
        for i in range(len(file_lines)):
            if _normalize_whitespace(file_lines[i]) == hint_norm:
                # Found hint line — try matching removals starting near here
                if hunk.removals:
                    norm_removals = [_normalize_whitespace(r) for r in hunk.removals]
                    # Search forward from hint location
                    for j in range(i, min(i + 20, len(file_lines))):
                        candidate = [
                            _normalize_whitespace(fl)
                            for fl in file_lines[j : j + len(hunk.removals)]
                        ]
                        if candidate == norm_removals:
                            # Adjust for context_before
                            start = j - len(hunk.context_before)
                            return max(0, start)
                elif hunk.additions:
                    # Pure addition — insert after context_before near hint
                    return i

    return None


def _apply_hunk(file_lines: list[str], hunk: Hunk) -> list[str]:
    """Apply a single hunk to *file_lines*, returning new lines."""
    loc = _find_hunk_location(file_lines, hunk)
    if loc is None:
        raise ValueError(
            "Could not locate hunk context in file. "
            f"Looking for: {hunk.context_before + hunk.removals}"
        )

    # Build replacement: context_before + additions + context_after
    replacement = hunk.context_before + hunk.additions + hunk.context_after
    span_len = len(hunk.context_before) + len(hunk.removals) + len(hunk.context_after)

    return file_lines[:loc] + replacement + file_lines[loc + span_len :]


async def apply_ops(ops: list[PatchOp], environment: ExecutionEnvironment) -> list[str]:
    """Apply a list of patch operations, return list of error messages."""
    errors: list[str] = []

    for op in ops:
        try:
            if op.kind == OpKind.ADD_FILE:
                await environment.write_file(op.path, op.content or "")

            elif op.kind == OpKind.DELETE_FILE:
                # Use rm to delete the file
                result = await environment.exec_command(f"rm -f {shlex.quote(op.path)}")
                if result.exit_code != 0:
                    errors.append(f"Failed to delete {op.path}: {result.stderr}")

            elif op.kind == OpKind.MOVE_FILE:
                # Read old, apply hunks, write new, delete old
                raw = await environment.read_file(op.path)
                content = _strip_line_numbers(raw)
                file_lines = content.splitlines()
                for hunk in op.hunks:
                    file_lines = _apply_hunk(file_lines, hunk)
                await environment.write_file(
                    op.new_path or op.path,
                    "\n".join(file_lines) + "\n" if file_lines else "",
                )
                if op.new_path and op.new_path != op.path:
                    await environment.exec_command(f"rm -f {shlex.quote(op.path)}")

            elif op.kind == OpKind.UPDATE_FILE:
                raw = await environment.read_file(op.path)
                content = _strip_line_numbers(raw)
                file_lines = content.splitlines()
                for hunk in op.hunks:
                    file_lines = _apply_hunk(file_lines, hunk)
                await environment.write_file(
                    op.path,
                    "\n".join(file_lines) + "\n" if file_lines else "",
                )

        except Exception as exc:
            errors.append(f"Error applying {op.kind.name} to {op.path}: {exc}")

    return errors


def _strip_line_numbers(text: str) -> str:
    """Remove cat -n style line numbers from read_file output."""
    lines = text.splitlines()
    stripped: list[str] = []
    for line in lines:
        tab_idx = line.find("\t")
        if tab_idx != -1:
            stripped.append(line[tab_idx + 1 :])
        else:
            stripped.append(line)
    return "\n".join(stripped)


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------


async def apply_patch(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Parse and apply a v4a patch."""
    patch_text: str = arguments["patch"]

    try:
        ops = parse_patch(patch_text)
    except Exception as exc:
        msg = f"Error parsing patch: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    if not ops:
        msg = "No operations found in patch."
        return ToolResult(output=msg, is_error=True, full_output=msg)

    errors = await apply_ops(ops, environment)
    if errors:
        msg = "Patch applied with errors:\n" + "\n".join(errors)
        return ToolResult(output=msg, is_error=True, full_output=msg)

    file_list = ", ".join(op.path for op in ops)
    msg = f"Patch applied successfully to: {file_list}"
    return ToolResult(output=msg, full_output=msg)


APPLY_PATCH_DEF = ToolDefinition(
    name="apply_patch",
    description=(
        "Apply a v4a-format patch to create, modify, delete, or move files. "
        "The patch must be wrapped in '*** Begin Patch' / '*** End Patch' markers. "
        "Supported operations: '*** Add File: <path>' (all lines prefixed with +), "
        "'*** Delete File: <path>', '*** Update File: <path>' with @@ hunks "
        "(space=context, -=remove, +=add), and optional '*** Move to: <path>' "
        "for renames. Show ~3 lines of context around each change."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": (
                    "The patch text in v4a format. Must start with "
                    "'*** Begin Patch' and end with '*** End Patch'."
                ),
            },
        },
        "required": ["patch"],
    },
)
