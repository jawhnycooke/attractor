"""Tests for core tools and tool registry against a temporary directory."""

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.tools.core_tools import (
    READ_FILE_DEF,
    WRITE_FILE_DEF,
    edit_file,
    glob_tool,
    grep_tool,
    list_dir_tool,
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


class TestListDir:
    @pytest.mark.asyncio
    async def test_list_dir_basic(self, env, tmp_path) -> None:
        (tmp_path / "file_a.txt").write_text("a")
        (tmp_path / "file_b.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = await list_dir_tool({"path": str(tmp_path)}, env)
        assert not result.is_error
        assert "file_a.txt" in result.output
        assert "file_b.py" in result.output
        assert "subdir/" in result.output

    @pytest.mark.asyncio
    async def test_list_dir_shows_sizes(self, env, tmp_path) -> None:
        (tmp_path / "sized.txt").write_text("hello world")
        result = await list_dir_tool({"path": str(tmp_path)}, env)
        assert not result.is_error
        assert "bytes" in result.output

    @pytest.mark.asyncio
    async def test_list_dir_empty(self, env, tmp_path) -> None:
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        result = await list_dir_tool({"path": str(empty)}, env)
        assert not result.is_error
        assert "empty directory" in result.output

    @pytest.mark.asyncio
    async def test_list_dir_with_depth(self, env, tmp_path) -> None:
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "nested.txt").write_text("nested")
        result = await list_dir_tool({"path": str(tmp_path), "depth": 2}, env)
        assert not result.is_error
        assert "nested.txt" in result.output

    @pytest.mark.asyncio
    async def test_list_dir_nonexistent(self, env) -> None:
        result = await list_dir_tool({"path": "/nonexistent/path/xyz"}, env)
        assert result.is_error
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_list_dir_default_depth(self, env, tmp_path) -> None:
        """Default depth=1 should not recurse into subdirectories."""
        (tmp_path / "top.txt").write_text("top")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_text("deep")
        result = await list_dir_tool({"path": str(tmp_path)}, env)
        assert not result.is_error
        assert "top.txt" in result.output
        assert "sub/" in result.output
        # deep.txt is at depth 2, should not appear with default depth=1
        assert "deep.txt" not in result.output


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


# ---------------------------------------------------------------------------
# Gap A-C04: edit_file replace_all parameter
# ---------------------------------------------------------------------------


class TestEditFileReplaceAll:
    @pytest.mark.asyncio
    async def test_replace_all_false_is_default(self, env, tmp_path) -> None:
        """Default behavior (replace_all absent) replaces single occurrence."""
        (tmp_path / "single.txt").write_text("hello world\n")
        result = await edit_file(
            {"path": "single.txt", "old_string": "hello", "new_string": "goodbye"},
            env,
        )
        assert not result.is_error
        assert (tmp_path / "single.txt").read_text() == "goodbye world\n"

    @pytest.mark.asyncio
    async def test_replace_all_false_errors_on_multiple(self, env, tmp_path) -> None:
        """replace_all=False errors when old_string appears multiple times."""
        (tmp_path / "multi.txt").write_text("aaa\naaa\naaa\n")
        result = await edit_file(
            {
                "path": "multi.txt",
                "old_string": "aaa",
                "new_string": "bbb",
                "replace_all": False,
            },
            env,
        )
        assert result.is_error
        assert "3 times" in result.output

    @pytest.mark.asyncio
    async def test_replace_all_true_replaces_all_occurrences(self, env, tmp_path) -> None:
        """replace_all=True replaces every occurrence of old_string."""
        (tmp_path / "multi.txt").write_text("aaa\naaa\naaa\n")
        result = await edit_file(
            {
                "path": "multi.txt",
                "old_string": "aaa",
                "new_string": "bbb",
                "replace_all": True,
            },
            env,
        )
        assert not result.is_error
        content = (tmp_path / "multi.txt").read_text()
        assert content == "bbb\nbbb\nbbb\n"
        assert "aaa" not in content
        assert "3 replacements" in result.output

    @pytest.mark.asyncio
    async def test_replace_all_true_single_occurrence(self, env, tmp_path) -> None:
        """replace_all=True with single occurrence works without error."""
        (tmp_path / "one.txt").write_text("find me here\n")
        result = await edit_file(
            {
                "path": "one.txt",
                "old_string": "find me",
                "new_string": "found you",
                "replace_all": True,
            },
            env,
        )
        assert not result.is_error
        assert (tmp_path / "one.txt").read_text() == "found you here\n"
        # Single replacement should not mention count
        assert "replacements" not in result.output

    @pytest.mark.asyncio
    async def test_replace_all_not_found(self, env, tmp_path) -> None:
        """replace_all=True still errors when old_string is absent."""
        (tmp_path / "empty_match.txt").write_text("nothing to match\n")
        result = await edit_file(
            {
                "path": "empty_match.txt",
                "old_string": "missing",
                "new_string": "replacement",
                "replace_all": True,
            },
            env,
        )
        assert result.is_error
        assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_replace_all_file_not_found(self, env) -> None:
        """replace_all=True on missing file returns error."""
        result = await edit_file(
            {
                "path": "nonexistent.txt",
                "old_string": "x",
                "new_string": "y",
                "replace_all": True,
            },
            env,
        )
        assert result.is_error
        assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_replace_all_adjacent_occurrences(self, env, tmp_path) -> None:
        """replace_all handles adjacent/overlapping-like patterns correctly."""
        (tmp_path / "adj.txt").write_text("ababab\n")
        result = await edit_file(
            {
                "path": "adj.txt",
                "old_string": "ab",
                "new_string": "xy",
                "replace_all": True,
            },
            env,
        )
        assert not result.is_error
        assert (tmp_path / "adj.txt").read_text() == "xyxyxy\n"
