"""Core tool implementations for the coding agent.

Each tool function accepts ``arguments`` (dict from the LLM) and
``environment`` (ExecutionEnvironment) and returns a ToolResult.
"""

from __future__ import annotations

from typing import Any

from attractor.agent.environment import ExecutionEnvironment
from attractor.agent.tools.registry import ToolResult
from attractor.llm.models import ToolDefinition

# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def read_file(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Read a file and return line-numbered content."""
    path: str = arguments["path"]
    offset = arguments.get("offset")
    limit = arguments.get("limit")

    if not await environment.file_exists(path):
        return ToolResult(
            output=f"Error: file not found: {path}",
            is_error=True,
            full_output=f"Error: file not found: {path}",
        )

    try:
        content = await environment.read_file(path, offset=offset, limit=limit)
    except Exception as exc:
        msg = f"Error reading {path}: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    return ToolResult(output=content, full_output=content)


READ_FILE_DEF = ToolDefinition(
    name="read_file",
    description=(
        "Read a file from the filesystem. Returns line-numbered content. "
        "Use offset and limit for large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
    },
)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def write_file(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Create or overwrite a file."""
    path: str = arguments["path"]
    content: str = arguments["content"]

    try:
        await environment.write_file(path, content)
    except Exception as exc:
        msg = f"Error writing {path}: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    msg = f"Successfully wrote to {path}"
    return ToolResult(output=msg, full_output=msg)


WRITE_FILE_DEF = ToolDefinition(
    name="write_file",
    description="Create or overwrite a file with the given content.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path for the file.",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
)


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


async def edit_file(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Search-and-replace edit within a file.

    Fails if old_string is not found or is not unique in the file.
    """
    path: str = arguments["path"]
    old_string: str = arguments["old_string"]
    new_string: str = arguments["new_string"]

    if not await environment.file_exists(path):
        msg = f"Error: file not found: {path}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    try:
        raw = await environment.read_file(path)
        # Strip line numbers from read_file output to get raw content
        lines = raw.splitlines(keepends=True)
        content_lines: list[str] = []
        for line in lines:
            # Line format: "     1\tcontent"
            tab_idx = line.find("\t")
            if tab_idx != -1:
                content_lines.append(line[tab_idx + 1 :])
            else:
                content_lines.append(line)
        content = "".join(content_lines)
    except Exception as exc:
        msg = f"Error reading {path}: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    count = content.count(old_string)
    if count == 0:
        msg = f"Error: old_string not found in {path}"
        return ToolResult(output=msg, is_error=True, full_output=msg)
    if count > 1:
        msg = (
            f"Error: old_string found {count} times in {path}. "
            f"Provide more context to make it unique."
        )
        return ToolResult(output=msg, is_error=True, full_output=msg)

    new_content = content.replace(old_string, new_string, 1)
    try:
        await environment.write_file(path, new_content)
    except Exception as exc:
        msg = f"Error writing {path}: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    msg = f"Successfully edited {path}"
    return ToolResult(output=msg, full_output=msg)


EDIT_FILE_DEF = ToolDefinition(
    name="edit_file",
    description=(
        "Make a search-and-replace edit in a file. The old_string must "
        "appear exactly once in the file. Provide enough context to be unique."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find (must be unique).",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


async def shell(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Execute a shell command."""
    command: str = arguments["command"]
    timeout_ms = arguments.get("timeout_ms", 10_000)

    result = await environment.exec_command(command, timeout_ms=timeout_ms)

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.timed_out:
        parts.append(f"[Command timed out after {timeout_ms}ms]")

    output = "\n".join(parts) if parts else "(no output)"
    is_error = result.exit_code != 0 or result.timed_out

    return ToolResult(output=output, is_error=is_error, full_output=output)


SHELL_DEF = ToolDefinition(
    name="shell",
    description=("Execute a shell command. Returns stdout, stderr, and exit code."),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 10000).",
            },
        },
        "required": ["command"],
    },
)


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


async def grep_tool(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """Regex search across files."""
    pattern: str = arguments["pattern"]
    path = arguments.get("path", environment.working_directory())
    include = arguments.get("include")
    max_results = arguments.get("max_results", 100)

    options: dict[str, str] = {"max_results": str(max_results)}
    if include:
        options["include"] = include

    try:
        output = await environment.grep(pattern, path, options=options)
    except Exception as exc:
        msg = f"Error searching: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    if not output:
        output = "No matches found."

    return ToolResult(output=output, full_output=output)


GREP_DEF = ToolDefinition(
    name="grep",
    description="Search for a regex pattern across files.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in.",
            },
            "include": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 100).",
            },
        },
        "required": ["pattern"],
    },
)


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


async def glob_tool(
    arguments: dict[str, Any],
    environment: ExecutionEnvironment,
) -> ToolResult:
    """File pattern matching sorted by modification time."""
    pattern: str = arguments["pattern"]
    path = arguments.get("path")

    try:
        results = await environment.glob(pattern, path=path)
    except Exception as exc:
        msg = f"Error globbing: {exc}"
        return ToolResult(output=msg, is_error=True, full_output=msg)

    if not results:
        output = "No files matched."
    else:
        output = "\n".join(results)

    return ToolResult(output=output, full_output=output)


GLOB_DEF = ToolDefinition(
    name="glob",
    description="Find files matching a glob pattern, sorted by modification time.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py').",
            },
            "path": {
                "type": "string",
                "description": "Base directory for the search.",
            },
        },
        "required": ["pattern"],
    },
)
