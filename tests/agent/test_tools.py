"""Tests for core tools and tool registry against a temporary directory."""

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.tools.core_tools import (
    READ_FILE_DEF,
    WRITE_FILE_DEF,
    edit_file,
    glob_tool,
    grep_tool,
    read_file,
    shell,
    write_file,
)
from attractor.agent.tools.registry import ToolRegistry


@pytest.fixture
async def env(tmp_path):
    """Create a LocalExecutionEnvironment rooted at tmp_path."""
    environment = LocalExecutionEnvironment(working_dir=str(tmp_path))
    await environment.initialize()
    yield environment
    await environment.cleanup()


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, env, tmp_path) -> None:
        (tmp_path / "hello.txt").write_text("line one\nline two\nline three\n")
        result = await read_file({"path": "hello.txt"}, env)
        assert not result.is_error
        assert "line one" in result.output
        assert "line two" in result.output

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, env, tmp_path) -> None:
        (tmp_path / "lines.txt").write_text("\n".join(f"L{i}" for i in range(20)))
        result = await read_file({"path": "lines.txt", "offset": 5, "limit": 3}, env)
        assert not result.is_error
        assert "L4" in result.output  # line 5 (0-indexed L4)
        assert "L0" not in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, env) -> None:
        result = await read_file({"path": "missing.txt"}, env)
        assert result.is_error
        assert "not found" in result.output


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, env, tmp_path) -> None:
        result = await write_file({"path": "new.txt", "content": "hello world"}, env)
        assert not result.is_error
        assert (tmp_path / "new.txt").read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, env, tmp_path) -> None:
        result = await write_file(
            {"path": "sub/dir/file.txt", "content": "nested"}, env
        )
        assert not result.is_error
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"


class TestEditFile:
    @pytest.mark.asyncio
    async def test_successful_edit(self, env, tmp_path) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    return 1\n")
        result = await edit_file(
            {
                "path": "code.py",
                "old_string": "return 1",
                "new_string": "return 42",
            },
            env,
        )
        assert not result.is_error
        assert "return 42" in (tmp_path / "code.py").read_text()

    @pytest.mark.asyncio
    async def test_edit_old_string_not_found(self, env, tmp_path) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    return 1\n")
        result = await edit_file(
            {
                "path": "code.py",
                "old_string": "nonexistent",
                "new_string": "replacement",
            },
            env,
        )
        assert result.is_error
        assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_edit_non_unique_old_string(self, env, tmp_path) -> None:
        (tmp_path / "code.py").write_text("x = 1\nx = 1\n")
        result = await edit_file(
            {"path": "code.py", "old_string": "x = 1", "new_string": "x = 2"},
            env,
        )
        assert result.is_error
        assert "2 times" in result.output


class TestShell:
    @pytest.mark.asyncio
    async def test_simple_command(self, env) -> None:
        result = await shell({"command": "echo hello"}, env)
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_command_failure(self, env) -> None:
        result = await shell({"command": "exit 1"}, env)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_command_timeout(self, env) -> None:
        result = await shell({"command": "sleep 10", "timeout_ms": 500}, env)
        assert result.is_error
        assert "timed out" in result.output.lower()


class TestGrep:
    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, env, tmp_path) -> None:
        (tmp_path / "search.txt").write_text("apple\nbanana\ncherry\n")
        result = await grep_tool({"pattern": "ban.*", "path": "search.txt"}, env)
        assert not result.is_error
        assert "banana" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_match(self, env, tmp_path) -> None:
        (tmp_path / "search.txt").write_text("apple\nbanana\ncherry\n")
        result = await grep_tool({"pattern": "xyz", "path": "search.txt"}, env)
        assert "No matches" in result.output


class TestGlob:
    @pytest.mark.asyncio
    async def test_glob_finds_files(self, env, tmp_path) -> None:
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        result = await glob_tool({"pattern": "*.py"}, env)
        assert not result.is_error
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output

    @pytest.mark.asyncio
    async def test_glob_no_match(self, env, tmp_path) -> None:
        result = await glob_tool({"pattern": "*.rs"}, env)
        assert "No files matched" in result.output


class TestToolRegistryUnregister:
    def test_unregister_removes_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("read_file", read_file, READ_FILE_DEF)
        assert registry.has_tool("read_file")
        assert any(d.name == "read_file" for d in registry.definitions())

        registry.unregister("read_file")
        assert not registry.has_tool("read_file")
        assert all(d.name != "read_file" for d in registry.definitions())

    def test_unregister_not_found_is_noop(self) -> None:
        registry = ToolRegistry()
        registry.register("write_file", write_file, WRITE_FILE_DEF)

        # Should not raise
        registry.unregister("nonexistent")
        assert registry.has_tool("write_file")

    def test_unregister_updates_definitions_and_names(self) -> None:
        registry = ToolRegistry()
        registry.register("read_file", read_file, READ_FILE_DEF)
        registry.register("write_file", write_file, WRITE_FILE_DEF)
        assert len(registry.definitions()) == 2
        assert set(registry.tool_names()) == {"read_file", "write_file"}

        registry.unregister("read_file")
        assert len(registry.definitions()) == 1
        assert registry.tool_names() == ["write_file"]

    @pytest.mark.asyncio
    async def test_dispatch_after_unregister_returns_error(self, env) -> None:
        registry = ToolRegistry()
        registry.register("read_file", read_file, READ_FILE_DEF)
        registry.unregister("read_file")

        result = await registry.dispatch("read_file", {"path": "x.txt"}, env)
        assert result.is_error
        assert "unknown tool" in result.output
