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
    apply_ops,
    parse_patch,
)

# ---------------------------------------------------------------------------
# parse_patch tests
# ---------------------------------------------------------------------------


class TestParseAddFile:
    def test_parse_add_file_extracts_content(self) -> None:
        patch = (
            "--- /dev/null\n"
            "+++ b/new_file.py\n"
            "@@ -0,0 +1,3 @@\n"
            "+line one\n"
            "+line two\n"
            "+line three\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.ADD_FILE
        assert op.path == "new_file.py"
        assert "line one" in op.content
        assert "line two" in op.content
        assert "line three" in op.content


class TestParseUpdateFile:
    def test_parse_update_file_multiple_hunks(self) -> None:
        patch = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            " first\n"
            "-old line\n"
            "+new line\n"
            " context\n"
            "@@ -10,3 +10,3 @@\n"
            " more context\n"
            "-another old\n"
            "+another new\n"
            " trailing\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.UPDATE_FILE
        assert op.path == "app.py"
        assert len(op.hunks) == 2


class TestParseMoveFile:
    def test_parse_move_file_captures_both_paths(self) -> None:
        patch = (
            "diff --git a/old_name.py b/new_name.py\n"
            "--- a/old_name.py\n"
            "+++ b/new_name.py\n"
            "@@ -1,3 +1,3 @@\n"
            " keep\n"
            "-remove\n"
            "+add\n"
            " keep2\n"
        )
        ops = parse_patch(patch)
        assert len(ops) == 1
        op = ops[0]
        assert op.kind == OpKind.MOVE_FILE
        assert op.path == "old_name.py"
        assert op.new_path == "new_name.py"


class TestParseEdgeCases:
    def test_parse_patch_empty_input(self) -> None:
        ops = parse_patch("")
        assert ops == []

    def test_parse_patch_malformed_no_hunk_header(self) -> None:
        # A diff header with no @@ hunk lines should not produce
        # an UPDATE_FILE with populated hunks.
        patch = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "some random line with no hunk\n"
        )
        ops = parse_patch(patch)
        # Should produce an op with no hunks
        assert len(ops) == 1
        assert ops[0].hunks == []


# ---------------------------------------------------------------------------
# _find_hunk_location tests
# ---------------------------------------------------------------------------


class TestFindHunkLocation:
    def test_find_hunk_location_with_duplicate_context(self) -> None:
        # Build a file where the same context appears twice.
        # The function should return the FIRST occurrence.
        file_lines = [
            "alpha",
            "beta",
            "gamma",
            "alpha",  # duplicate at index 3
            "beta",
            "gamma",
            "extra",
        ]
        hunk = Hunk(
            context_before=["alpha", "beta"],
            removals=["gamma"],
        )
        loc = _find_hunk_location(file_lines, hunk)
        assert loc == 0  # first occurrence

    def test_find_hunk_location_no_match_raises(self) -> None:
        file_lines = ["aaa", "bbb", "ccc"]
        hunk = Hunk(
            context_before=["zzz"],
            removals=["yyy"],
        )
        loc = _find_hunk_location(file_lines, hunk)
        assert loc is None
        # _apply_hunk wraps this in a ValueError
        with pytest.raises(ValueError, match="Could not locate hunk context"):
            _apply_hunk(file_lines, hunk)


# ---------------------------------------------------------------------------
# apply_ops tests
# ---------------------------------------------------------------------------


class TestApplyOps:
    @pytest.mark.asyncio
    async def test_apply_ops_delete_uses_shlex_quote(self) -> None:
        """Regression test for B1: paths with special chars must be quoted."""
        env = MagicMock()
        env.exec_command = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", exit_code=0)
        )

        path_with_quotes = "file's name.py"
        op = PatchOp(kind=OpKind.DELETE_FILE, path=path_with_quotes)

        errors = await apply_ops([op], env)
        assert errors == []

        # Verify the command uses shlex.quote
        call_args = env.exec_command.call_args[0][0]
        expected_quoted = shlex.quote(path_with_quotes)
        assert expected_quoted in call_args

    @pytest.mark.asyncio
    async def test_apply_ops_move_applies_hunks_then_writes(self, tmp_path) -> None:
        env = MagicMock()
        _old_content = "keep\nremove_me\nkeep2\n"  # noqa: F841
        # read_file returns numbered lines (cat -n style)
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

        # Should have written to new path
        env.write_file.assert_called_once()
        written_path = env.write_file.call_args[0][0]
        written_content = env.write_file.call_args[0][1]
        assert written_path == "new.py"
        assert "added" in written_content
        assert "remove_me" not in written_content

        # Should have deleted old path
        env.exec_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_ops_update_sequential_hunks(self) -> None:
        env = MagicMock()
        # 10 lines: line0 through line9
        lines = [f"line{i}" for i in range(10)]
        numbered = "\n".join(f"{i+1:>6}\t{line}" for i, line in enumerate(lines)) + "\n"
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
