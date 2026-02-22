"""Tests for the apply_patch tool — v4a format parser and applicator."""

import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor.agent.environment import ExecResult
from attractor.agent.tools.apply_patch import (
    Hunk,
    OpKind,
    PatchOp,
    _apply_hunk,
    _find_hunk_location,
    _normalize_whitespace,
    apply_ops,
    apply_patch,
    parse_patch,
)

# ---------------------------------------------------------------------------
# parse_patch — Add File
# ---------------------------------------------------------------------------


class TestParseAddFile:
    def test_add_file_basic(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: src/utils/helpers.py\n"
            "+def greet(name):\n"
            '+    return f"Hello, {name}!"\n'
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.ADD_FILE
        assert op.path == "src/utils/helpers.py"
        assert op.content is not None
        assert "def greet(name):" in op.content
        assert 'return f"Hello, {name}!"' in op.content

    def test_add_file_empty_content(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: src/__init__.py\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].kind == OpKind.ADD_FILE
        assert ops[0].path == "src/__init__.py"
        assert ops[0].content == ""

    def test_add_file_multiline(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: new.py\n"
            "+line one\n"
            "+line two\n"
            "+line three\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].content == "line one\nline two\nline three\n"


# ---------------------------------------------------------------------------
# parse_patch — Delete File
# ---------------------------------------------------------------------------


class TestParseDeleteFile:
    def test_delete_file_basic(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: src/old_module.py\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].kind == OpKind.DELETE_FILE
        assert ops[0].path == "src/old_module.py"

    def test_delete_file_no_content(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: obsolete.py\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].content is None
        assert ops[0].hunks == []


# ---------------------------------------------------------------------------
# parse_patch — Update File
# ---------------------------------------------------------------------------


class TestParseUpdateFile:
    def test_update_file_single_hunk(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: src/main.py\n"
            "@@ def main():\n"
            '     print("Hello")\n'
            "-    return 0\n"
            '+    print("World")\n'
            "+    return 1\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.UPDATE_FILE
        assert op.path == "src/main.py"
        assert len(op.hunks) == 1
        hunk = op.hunks[0]
        assert hunk.context_hint == "def main():"
        assert hunk.context_before == ['    print("Hello")']
        assert hunk.removals == ["    return 0"]
        assert hunk.additions == ['    print("World")', "    return 1"]

    def test_update_file_multi_hunk(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: src/config.py\n"
            "@@ DEFAULT_TIMEOUT = 30\n"
            "-DEFAULT_TIMEOUT = 30\n"
            "+DEFAULT_TIMEOUT = 60\n"
            "@@ def load_config():\n"
            "     config = {}\n"
            '-    config["debug"] = False\n'
            '+    config["debug"] = True\n'
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert len(ops[0].hunks) == 2

        hunk1 = ops[0].hunks[0]
        assert hunk1.context_hint == "DEFAULT_TIMEOUT = 30"
        assert hunk1.removals == ["DEFAULT_TIMEOUT = 30"]
        assert hunk1.additions == ["DEFAULT_TIMEOUT = 60"]

        hunk2 = ops[0].hunks[1]
        assert hunk2.context_hint == "def load_config():"
        assert hunk2.context_before == ["    config = {}"]
        assert hunk2.removals == ['    config["debug"] = False']
        assert hunk2.additions == ['    config["debug"] = True']

    def test_update_file_with_context_after(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: app.py\n"
            "@@ first line\n"
            " first\n"
            "-old line\n"
            "+new line\n"
            " context_after\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        hunk = ops[0].hunks[0]
        assert hunk.context_before == ["first"]
        assert hunk.removals == ["old line"]
        assert hunk.additions == ["new line"]
        assert hunk.context_after == ["context_after"]


# ---------------------------------------------------------------------------
# parse_patch — Move File (Update + Rename)
# ---------------------------------------------------------------------------


class TestParseMoveFile:
    def test_move_file_with_hunks(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: old_name.py\n"
            "*** Move to: new_name.py\n"
            "@@ import os\n"
            " import sys\n"
            "-import old_dep\n"
            "+import new_dep\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.MOVE_FILE
        assert op.path == "old_name.py"
        assert op.new_path == "new_name.py"
        assert len(op.hunks) == 1
        assert op.hunks[0].removals == ["import old_dep"]
        assert op.hunks[0].additions == ["import new_dep"]

    def test_move_file_no_hunks(self) -> None:
        """A rename without content changes should still produce MOVE_FILE."""
        patch = (
            "*** Begin Patch\n"
            "*** Update File: old.py\n"
            "*** Move to: new.py\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].kind == OpKind.MOVE_FILE
        assert ops[0].path == "old.py"
        assert ops[0].new_path == "new.py"


# ---------------------------------------------------------------------------
# parse_patch — Edge Cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    def test_empty_input(self) -> None:
        ops = parse_patch("")
        assert ops == []

    def test_multiple_operations(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: new.py\n"
            "+content\n"
            "*** Delete File: old.py\n"
            "*** Update File: main.py\n"
            "@@ start\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 3
        assert ops[0].kind == OpKind.ADD_FILE
        assert ops[1].kind == OpKind.DELETE_FILE
        assert ops[2].kind == OpKind.UPDATE_FILE

    def test_eof_marker_terminates_hunk(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: main.py\n"
            "@@ end of file\n"
            " last_line\n"
            "+added_line\n"
            "*** End of File\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert len(ops[0].hunks) == 1
        assert ops[0].hunks[0].additions == ["added_line"]

    def test_no_begin_patch_marker(self) -> None:
        """Parser should handle v4a content without explicit Begin marker."""
        patch = (
            "*** Add File: no_begin.py\n"
            "+content\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].kind == OpKind.ADD_FILE


# ---------------------------------------------------------------------------
# _find_hunk_location tests
# ---------------------------------------------------------------------------


class TestFindHunkLocation:
    def test_exact_match(self) -> None:
        file_lines = ["alpha", "beta", "gamma", "delta"]
        hunk = Hunk(context_before=["alpha", "beta"], removals=["gamma"])
        loc = _find_hunk_location(file_lines, hunk)
        assert loc == 0

    def test_duplicate_context_returns_first(self) -> None:
        file_lines = [
            "alpha",
            "beta",
            "gamma",
            "alpha",  # duplicate at index 3
            "beta",
            "gamma",
            "extra",
        ]
        hunk = Hunk(context_before=["alpha", "beta"], removals=["gamma"])
        loc = _find_hunk_location(file_lines, hunk)
        assert loc == 0

    def test_no_match_returns_none(self) -> None:
        file_lines = ["aaa", "bbb", "ccc"]
        hunk = Hunk(context_before=["zzz"], removals=["yyy"])
        loc = _find_hunk_location(file_lines, hunk)
        assert loc is None

    def test_apply_hunk_raises_on_no_match(self) -> None:
        file_lines = ["aaa", "bbb", "ccc"]
        hunk = Hunk(context_before=["zzz"], removals=["yyy"])
        with pytest.raises(ValueError, match="Could not locate hunk context"):
            _apply_hunk(file_lines, hunk)

    def test_fuzzy_match_whitespace_normalization(self) -> None:
        file_lines = ["def  foo(  x  ):", "    return x"]
        hunk = Hunk(
            context_before=[],
            removals=["def foo( x ):"],
            additions=["def foo(x, y):"],
        )
        loc = _find_hunk_location(file_lines, hunk)
        assert loc == 0

    def test_empty_search_returns_zero(self) -> None:
        file_lines = ["a", "b", "c"]
        hunk = Hunk()
        loc = _find_hunk_location(file_lines, hunk)
        assert loc == 0


# ---------------------------------------------------------------------------
# _normalize_whitespace tests
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_collapses_spaces(self) -> None:
        assert _normalize_whitespace("def  foo(  x  ):") == "def foo( x ):"

    def test_strips_leading_trailing(self) -> None:
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_tabs_and_spaces(self) -> None:
        assert _normalize_whitespace("a\t\tb") == "a b"


# ---------------------------------------------------------------------------
# apply_ops tests
# ---------------------------------------------------------------------------


class TestApplyOps:
    @pytest.mark.asyncio
    async def test_add_file(self) -> None:
        env = MagicMock()
        env.write_file = AsyncMock()

        op = PatchOp(kind=OpKind.ADD_FILE, path="new.py", content="hello\n")
        errors = await apply_ops([op], env)
        assert errors == []
        env.write_file.assert_called_once_with("new.py", "hello\n")

    @pytest.mark.asyncio
    async def test_delete_uses_shlex_quote(self) -> None:
        """Paths with special chars must be quoted."""
        env = MagicMock()
        env.exec_command = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", exit_code=0)
        )

        path_with_quotes = "file's name.py"
        op = PatchOp(kind=OpKind.DELETE_FILE, path=path_with_quotes)

        errors = await apply_ops([op], env)
        assert errors == []

        call_args = env.exec_command.call_args[0][0]
        expected_quoted = shlex.quote(path_with_quotes)
        assert expected_quoted in call_args

    @pytest.mark.asyncio
    async def test_move_applies_hunks_then_writes(self) -> None:
        env = MagicMock()
        numbered = "     1\tkeep\n     2\tremove_me\n     3\tkeep2\n"
        env.read_file = AsyncMock(return_value=numbered)
        env.write_file = AsyncMock()
        env.exec_command = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", exit_code=0)
        )

        hunk = Hunk(
            context_before=["keep"],
            removals=["remove_me"],
            additions=["added"],
            context_after=["keep2"],
        )
        op = PatchOp(
            kind=OpKind.MOVE_FILE,
            path="old.py",
            new_path="new.py",
            hunks=[hunk],
        )

        errors = await apply_ops([op], env)
        assert errors == []

        env.write_file.assert_called_once()
        written_path = env.write_file.call_args[0][0]
        written_content = env.write_file.call_args[0][1]
        assert written_path == "new.py"
        assert "added" in written_content
        assert "remove_me" not in written_content

        # Should have deleted old path
        env.exec_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_sequential_hunks(self) -> None:
        env = MagicMock()
        lines = [f"line{i}" for i in range(10)]
        numbered = (
            "\n".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(lines)) + "\n"
        )
        env.read_file = AsyncMock(return_value=numbered)
        env.write_file = AsyncMock()

        hunk1 = Hunk(
            context_before=["line0"],
            removals=["line1"],
            additions=["replaced1"],
            context_after=["line2"],
        )
        hunk2 = Hunk(
            context_before=["line7"],
            removals=["line8"],
            additions=["replaced8"],
            context_after=["line9"],
        )
        op = PatchOp(
            kind=OpKind.UPDATE_FILE,
            path="test.py",
            hunks=[hunk1, hunk2],
        )

        errors = await apply_ops([op], env)
        assert errors == []

        written = env.write_file.call_args[0][1]
        assert "replaced1" in written
        assert "replaced8" in written
        assert "line1" not in written
        assert "line8" not in written

    @pytest.mark.asyncio
    async def test_delete_failure_reports_error(self) -> None:
        env = MagicMock()
        env.exec_command = AsyncMock(
            return_value=ExecResult(stdout="", stderr="permission denied", exit_code=1)
        )
        op = PatchOp(kind=OpKind.DELETE_FILE, path="locked.py")

        errors = await apply_ops([op], env)
        assert len(errors) == 1
        assert "Failed to delete" in errors[0]


# ---------------------------------------------------------------------------
# apply_patch (tool interface) tests
# ---------------------------------------------------------------------------


class TestApplyPatchTool:
    @pytest.mark.asyncio
    async def test_successful_patch(self) -> None:
        env = MagicMock()
        env.write_file = AsyncMock()

        result = await apply_patch(
            {
                "patch": (
                    "*** Begin Patch\n"
                    "*** Add File: hello.py\n"
                    "+print('hello')\n"
                    "*** End Patch\n"
                )
            },
            env,
        )
        assert not result.is_error
        assert "hello.py" in result.output

    @pytest.mark.asyncio
    async def test_empty_patch_returns_error(self) -> None:
        env = MagicMock()
        result = await apply_patch({"patch": ""}, env)
        assert result.is_error
        assert "No operations found" in result.output

    @pytest.mark.asyncio
    async def test_patch_with_apply_error(self) -> None:
        env = MagicMock()
        env.read_file = AsyncMock(side_effect=FileNotFoundError("not found"))

        result = await apply_patch(
            {
                "patch": (
                    "*** Begin Patch\n"
                    "*** Update File: missing.py\n"
                    "@@ def foo():\n"
                    "-old\n"
                    "+new\n"
                    "*** End Patch\n"
                )
            },
            env,
        )
        assert result.is_error
        assert "errors" in result.output.lower()


# ---------------------------------------------------------------------------
# End-to-end parse + apply tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_add_file_flow(self) -> None:
        """Parse v4a add file and apply it."""
        env = MagicMock()
        env.write_file = AsyncMock()

        patch = (
            "*** Begin Patch\n"
            "*** Add File: src/new_module.py\n"
            "+import os\n"
            "+\n"
            "+def main():\n"
            "+    pass\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        assert ops[0].kind == OpKind.ADD_FILE

        errors = await apply_ops(ops, env)
        assert errors == []
        env.write_file.assert_called_once()
        written = env.write_file.call_args[0][1]
        assert "import os" in written
        assert "def main():" in written

    @pytest.mark.asyncio
    async def test_full_update_file_flow(self) -> None:
        """Parse v4a update file and apply it to file content."""
        env = MagicMock()
        file_content = (
            "     1\timport os\n"
            "     2\t\n"
            "     3\tdef main():\n"
            '     4\t    print("Hello")\n'
            "     5\t    return 0\n"
        )
        env.read_file = AsyncMock(return_value=file_content)
        env.write_file = AsyncMock()

        patch = (
            "*** Begin Patch\n"
            "*** Update File: main.py\n"
            "@@ def main():\n"
            '     print("Hello")\n'
            "-    return 0\n"
            "+    return 1\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        errors = await apply_ops(ops, env)
        assert errors == []

        written = env.write_file.call_args[0][1]
        assert "return 1" in written
        assert "return 0" not in written
        assert 'print("Hello")' in written

    @pytest.mark.asyncio
    async def test_full_delete_then_add(self) -> None:
        """Multiple operations in one patch."""
        env = MagicMock()
        env.exec_command = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", exit_code=0)
        )
        env.write_file = AsyncMock()

        patch = (
            "*** Begin Patch\n"
            "*** Delete File: old.py\n"
            "*** Add File: new.py\n"
            "+replacement\n"
            "*** End Patch\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 2

        errors = await apply_ops(ops, env)
        assert errors == []
