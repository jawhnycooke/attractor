"""Tests for LocalExecutionEnvironment."""

import pytest

from attractor.agent.environment import LocalExecutionEnvironment


@pytest.fixture
async def env(tmp_path):
    """Create a LocalExecutionEnvironment rooted at tmp_path."""
    environment = LocalExecutionEnvironment(working_dir=str(tmp_path))
    await environment.initialize()
    yield environment
    await environment.cleanup()


class TestFileOperations:
    @pytest.mark.asyncio
    async def test_write_and_read(self, env, tmp_path) -> None:
        await env.write_file("test.txt", "hello\nworld\n")
        content = await env.read_file("test.txt")
        assert "hello" in content
        assert "world" in content

    @pytest.mark.asyncio
    async def test_read_with_line_numbers(self, env, tmp_path) -> None:
        await env.write_file("numbered.txt", "first\nsecond\nthird\n")
        content = await env.read_file("numbered.txt")
        # Should contain line numbers in cat -n format
        assert "\t" in content  # tab between number and content

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, env) -> None:
        lines = "\n".join(f"line{i}" for i in range(20)) + "\n"
        await env.write_file("many.txt", lines)
        content = await env.read_file("many.txt", offset=5, limit=3)
        assert "line4" in content
        assert "line0" not in content

    @pytest.mark.asyncio
    async def test_file_exists(self, env, tmp_path) -> None:
        (tmp_path / "exists.txt").write_text("x")
        assert await env.file_exists("exists.txt")
        assert not await env.file_exists("missing.txt")

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, env, tmp_path) -> None:
        await env.write_file("deep/nested/file.txt", "content")
        assert (tmp_path / "deep" / "nested" / "file.txt").exists()


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_directory(self, env, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "b.txt").write_text("b")

        entries = await env.list_directory(str(tmp_path))
        names = {e.name for e in entries}
        assert "a.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_directory_depth(self, env, tmp_path) -> None:
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "dir2").mkdir()
        (tmp_path / "dir1" / "dir2" / "deep.txt").write_text("deep")

        entries = await env.list_directory(str(tmp_path), depth=1)
        paths = {e.path for e in entries}
        # depth=1 should only show immediate children
        assert str(tmp_path / "dir1") in paths
        # dir2 is at depth 0 from dir1, which is at depth 1
        # Since we recurse into dirs with depth-1, dir2 content won't show
        deep_paths = [e for e in entries if "deep.txt" in e.name]
        assert len(deep_paths) == 0


class TestExecCommand:
    @pytest.mark.asyncio
    async def test_simple_command(self, env) -> None:
        result = await env.exec_command("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_command_failure(self, env) -> None:
        result = await env.exec_command("exit 1")
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_command_timeout(self, env) -> None:
        result = await env.exec_command("sleep 10", timeout_ms=500)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_working_directory(self, env, tmp_path) -> None:
        result = await env.exec_command("pwd")
        assert str(tmp_path) in result.stdout


class TestGrep:
    @pytest.mark.asyncio
    async def test_grep_file(self, env, tmp_path) -> None:
        (tmp_path / "data.txt").write_text("foo bar\nbaz qux\nfoo again\n")
        result = await env.grep("foo", str(tmp_path / "data.txt"))
        assert "foo bar" in result
        assert "foo again" in result

    @pytest.mark.asyncio
    async def test_grep_directory(self, env, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("hello world\n")
        (tmp_path / "b.txt").write_text("hello again\n")
        result = await env.grep("hello", str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result


class TestGlob:
    @pytest.mark.asyncio
    async def test_glob_pattern(self, env, tmp_path) -> None:
        (tmp_path / "x.py").write_text("")
        (tmp_path / "y.py").write_text("")
        (tmp_path / "z.txt").write_text("")
        results = await env.glob("*.py")
        py_files = [r for r in results if r.endswith(".py")]
        assert len(py_files) == 2


class TestPlatform:
    def test_platform_returns_string(self) -> None:
        env = LocalExecutionEnvironment()
        assert isinstance(env.platform(), str)
        assert len(env.platform()) > 0

    def test_working_directory_returns_string(self) -> None:
        env = LocalExecutionEnvironment()
        assert isinstance(env.working_directory(), str)
