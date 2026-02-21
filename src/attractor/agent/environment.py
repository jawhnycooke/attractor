"""Execution environment abstraction and local implementation.

Provides a Protocol defining the environment interface that tools operate
against, plus a LocalExecutionEnvironment that runs on the host machine.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DirEntry:
    """A single entry returned by list_directory."""

    name: str
    path: str
    is_dir: bool
    size: int = 0


@dataclass
class ExecResult:
    """Result of a command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


# Patterns for environment variable names that should be filtered out.
_FILTERED_ENV_PATTERNS: list[str] = [
    "*_API_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*_PASSWORD",
    "AWS_*KEY*",
    "DATABASE_URL",
    "*_DATABASE_URL",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "NPM_TOKEN",
    "DOCKER_*",
]


def _filter_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *env* (or os.environ) with sensitive vars removed."""
    source = dict(env) if env is not None else dict(os.environ)
    filtered: dict[str, str] = {}
    for key, value in source.items():
        upper_key = key.upper()
        if any(fnmatch.fnmatch(upper_key, pat) for pat in _FILTERED_ENV_PATTERNS):
            continue
        filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Interface that agent tools use to interact with the outside world."""

    async def read_file(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Read a file and return its contents with line numbers."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating parent directories as needed."""
        ...

    async def file_exists(self, path: str) -> bool:
        """Return True if the file at *path* exists."""
        ...

    async def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        """List directory entries up to *depth* levels deep."""
        ...

    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a shell command and return the result."""
        ...

    async def grep(
        self,
        pattern: str,
        path: str,
        options: dict[str, str] | None = None,
    ) -> str:
        """Search for *pattern* in files under *path*."""
        ...

    async def glob(self, pattern: str, path: str | None = None) -> list[str]:
        """Return file paths matching the glob *pattern*."""
        ...

    async def initialize(self) -> None:
        """Perform any setup needed before tool execution."""
        ...

    async def cleanup(self) -> None:
        """Release resources when the session ends."""
        ...

    def working_directory(self) -> str:
        """Return the current working directory path."""
        ...

    def platform(self) -> str:
        """Return the platform identifier (e.g. 'linux', 'darwin')."""
        ...


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------


class LocalExecutionEnvironment:
    """ExecutionEnvironment backed by the host filesystem and OS."""

    def __init__(self, working_dir: str | None = None) -> None:
        self._working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.subagents: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        """Create the working directory if it does not exist."""
        self._working_dir.mkdir(parents=True, exist_ok=True)

    async def cleanup(self) -> None:
        """No-op for local environment — nothing to tear down."""

    def working_directory(self) -> str:
        """Return the absolute path of the working directory."""
        return str(self._working_dir)

    def platform(self) -> str:
        """Return the current platform identifier."""
        return sys.platform

    # -- file operations ----------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve *path* against the working directory.

        Absolute paths are returned as-is; relative paths are joined
        with the working directory, then resolved to a canonical form.
        """
        p = Path(path)
        if not p.is_absolute():
            p = self._working_dir / p
        return p.resolve()

    async def read_file(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Read a file and return numbered lines (cat -n style)."""
        resolved = self._resolve(path)
        text = await asyncio.to_thread(resolved.read_text, encoding="utf-8")
        lines = text.splitlines(keepends=True)

        start = (offset or 1) - 1  # 1-based offset
        if start < 0:
            start = 0
        end = start + limit if limit else len(lines)

        numbered: list[str] = []
        for idx, line in enumerate(lines[start:end], start=start + 1):
            numbered.append(f"{idx:>6}\t{line}")
        return "".join(numbered)

    async def write_file(self, path: str, content: str) -> None:
        """Write *content* to the file at *path*, creating parents as needed."""
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")

    async def file_exists(self, path: str) -> bool:
        """Return True if the file at *path* exists on disk."""
        return await asyncio.to_thread(self._resolve(path).exists)

    async def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        """List directory entries up to *depth* levels deep."""
        resolved = self._resolve(path)
        entries: list[DirEntry] = []
        await asyncio.to_thread(self._walk_dir, resolved, resolved, depth, entries)
        return entries

    def _walk_dir(
        self,
        base: Path,
        current: Path,
        depth: int,
        out: list[DirEntry],
    ) -> None:
        if depth <= 0:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return
        for child in children:
            is_dir = child.is_dir()
            size = child.stat().st_size if not is_dir else 0
            out.append(
                DirEntry(
                    name=child.name,
                    path=str(child),
                    is_dir=is_dir,
                    size=size,
                )
            )
            if is_dir:
                self._walk_dir(base, child, depth - 1, out)

    # -- command execution --------------------------------------------------

    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a shell command with timeout and signal handling."""
        cwd = working_dir or str(self._working_dir)
        env = _filter_env(env_vars)

        timeout_s = timeout_ms / 1000.0

        def _run() -> ExecResult:
            kwargs: dict[str, Any] = dict(
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            proc = subprocess.Popen(command, **kwargs)
            timed_out = False
            try:
                stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Graceful: SIGTERM to process group
                pgid = os.getpgid(proc.pid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    stdout_b, stderr_b = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    # Hard kill
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout_b, stderr_b = proc.communicate()

            return ExecResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
                timed_out=timed_out,
            )

        return await asyncio.to_thread(_run)

    # -- search tools -------------------------------------------------------

    async def grep(
        self,
        pattern: str,
        path: str,
        options: dict[str, str] | None = None,
    ) -> str:
        """Search for *pattern* in files under *path*, returning matches."""
        resolved = self._resolve(path)
        opts = options or {}
        include = opts.get("include")
        max_results = int(opts.get("max_results", "200"))

        regex = re.compile(pattern)
        matches: list[str] = []

        def _search() -> None:
            targets: list[Path] = []
            if resolved.is_file():
                targets.append(resolved)
            else:
                glob_pat = include or "**/*"
                targets = [p for p in resolved.glob(glob_pat) if p.is_file()]
            for fpath in targets:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except (PermissionError, OSError):
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = str(fpath.relative_to(self._working_dir))
                        matches.append(f"{rel}:{i}: {line}")
                        if len(matches) >= max_results:
                            return

        await asyncio.to_thread(_search)
        return "\n".join(matches)

    async def glob(self, pattern: str, path: str | None = None) -> list[str]:
        """Return file paths matching *pattern*, sorted by modification time."""
        base = self._resolve(path) if path else self._working_dir

        def _glob() -> list[str]:
            results = sorted(
                base.glob(pattern),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            return [str(p) for p in results]

        return await asyncio.to_thread(_glob)
