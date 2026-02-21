"""apply_patch tool — v4a format parser for the OpenAI codex profile.

Parses a unified-diff-like patch format into discrete operations
(AddFile, DeleteFile, UpdateFile, MoveFile) and applies them to the
filesystem via the ExecutionEnvironment.
"""

from __future__ import annotations

import re
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
# Parser
# ---------------------------------------------------------------------------

_FILE_HEADER_RE = re.compile(
    r"^---\s+(?:a/)?(.*?)\s*$|"
    r"^\+\+\+\s+(?:b/)?(.*?)\s*$|"
    r"^diff\s+--git\s+a/(.*?)\s+b/(.*?)\s*$"
)
_HUNK_HEADER_RE = re.compile(r"^@@\s.*?@@")


def parse_patch(text: str) -> list[PatchOp]:
    """Parse v4a-style patch text into a list of PatchOp."""
    ops: list[PatchOp] = []
    lines = text.splitlines(keepends=True)
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n\r")

        # --- new file ---
        if line.startswith("--- /dev/null"):
            i += 1
            if i < len(lines) and lines[i].rstrip("\n\r").startswith("+++ "):
                path = lines[i].rstrip("\n\r").split("+++ ", 1)[1]
                path = re.sub(r"^b/", "", path).strip()
                i += 1
                # skip hunk header
                if i < len(lines) and _HUNK_HEADER_RE.match(
                    lines[i].rstrip("\n\r")
                ):
                    i += 1
                content_lines: list[str] = []
                while i < len(lines):
                    l = lines[i].rstrip("\n\r")
                    if l.startswith("diff ") or l.startswith("--- "):
                        break
                    if l.startswith("+"):
                        content_lines.append(l[1:])
                    i += 1
                ops.append(PatchOp(
                    kind=OpKind.ADD_FILE,
                    path=path,
                    content="\n".join(content_lines)
                    + ("\n" if content_lines else ""),
                ))
            continue

        # --- delete file ---
        if line.startswith("+++ /dev/null"):
            # previous line should have been --- a/path
            if ops or (i > 0 and lines[i - 1].rstrip("\n\r").startswith("--- ")):
                prev = lines[i - 1].rstrip("\n\r")
                path = re.sub(r"^---\s+a/", "", prev).strip()
                ops.append(PatchOp(kind=OpKind.DELETE_FILE, path=path))
            i += 1
            # skip remaining hunk lines for the delete
            while i < len(lines):
                l = lines[i].rstrip("\n\r")
                if l.startswith("diff ") or l.startswith("--- "):
                    break
                i += 1
            continue

        # --- diff header for update/move ---
        diff_match = re.match(
            r"^diff\s+--git\s+a/(.*?)\s+b/(.*?)\s*$", line
        )
        if diff_match:
            old_path = diff_match.group(1)
            new_path = diff_match.group(2)
            i += 1

            # skip --- and +++ lines
            while i < len(lines):
                l = lines[i].rstrip("\n\r")
                if l.startswith("--- ") or l.startswith("+++ "):
                    i += 1
                else:
                    break

            # parse hunks
            hunks: list[Hunk] = []
            while i < len(lines):
                l = lines[i].rstrip("\n\r")
                if l.startswith("diff "):
                    break
                if _HUNK_HEADER_RE.match(l):
                    i += 1
                    hunk = Hunk()
                    in_changes = False
                    while i < len(lines):
                        hl = lines[i].rstrip("\n\r")
                        if (
                            hl.startswith("diff ")
                            or _HUNK_HEADER_RE.match(hl)
                        ):
                            break
                        if hl.startswith("-"):
                            hunk.removals.append(hl[1:])
                            in_changes = True
                        elif hl.startswith("+"):
                            hunk.additions.append(hl[1:])
                            in_changes = True
                        elif hl.startswith(" "):
                            if in_changes:
                                hunk.context_after.append(hl[1:])
                            else:
                                hunk.context_before.append(hl[1:])
                        i += 1
                    hunks.append(hunk)
                else:
                    i += 1

            if old_path != new_path:
                ops.append(PatchOp(
                    kind=OpKind.MOVE_FILE,
                    path=old_path,
                    new_path=new_path,
                    hunks=hunks,
                ))
            else:
                ops.append(PatchOp(
                    kind=OpKind.UPDATE_FILE,
                    path=old_path,
                    hunks=hunks,
                ))
            continue

        i += 1

    return ops


# ---------------------------------------------------------------------------
# Applicator
# ---------------------------------------------------------------------------

def _find_hunk_location(
    file_lines: list[str], hunk: Hunk
) -> int | None:
    """Find where a hunk's context lines match in the file.

    Returns the index of the first line of the context_before match,
    or the first line of the removals if no context_before.
    """
    search_lines = hunk.context_before + hunk.removals
    if not search_lines:
        return 0

    for i in range(len(file_lines)):
        if file_lines[i: i + len(search_lines)] == search_lines:
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
    span_len = (
        len(hunk.context_before)
        + len(hunk.removals)
        + len(hunk.context_after)
    )

    return file_lines[:loc] + replacement + file_lines[loc + span_len:]


async def apply_ops(
    ops: list[PatchOp], environment: ExecutionEnvironment
) -> list[str]:
    """Apply a list of patch operations, return list of error messages."""
    errors: list[str] = []

    for op in ops:
        try:
            if op.kind == OpKind.ADD_FILE:
                await environment.write_file(op.path, op.content or "")

            elif op.kind == OpKind.DELETE_FILE:
                # Write empty to effectively "delete" — real delete would need
                # an env method; for now we clear it.
                result = await environment.exec_command(f"rm -f '{op.path}'")
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
                    await environment.exec_command(f"rm -f '{op.path}'")

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
            stripped.append(line[tab_idx + 1:])
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
        "Apply a v4a-format patch to create, modify, delete, or move files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "The patch text in v4a unified diff format.",
            },
        },
        "required": ["patch"],
    },
)
