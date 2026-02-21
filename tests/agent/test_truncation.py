"""Tests for the two-stage truncation pipeline."""

from attractor.agent.truncation import (
    TruncationConfig,
    truncate_output,
)


class TestCharacterTruncation:
    """Stage 1 — character-based truncation."""

    def test_short_output_unchanged(self) -> None:
        truncated, full = truncate_output("read_file", "hello world")
        assert truncated == "hello world"
        assert full == "hello world"

    def test_long_output_truncated(self) -> None:
        text = "x" * 100_000
        truncated, full = truncate_output("read_file", text)
        assert len(full) == 100_000
        assert len(truncated) < len(full)
        assert "characters removed from middle of output" in truncated

    def test_head_and_tail_preserved(self) -> None:
        head = "HEAD_MARKER_" + "a" * 1000
        tail = "b" * 1000 + "_TAIL_MARKER"
        middle = "m" * 100_000
        text = head + middle + tail
        truncated, _ = truncate_output("read_file", text)
        assert truncated.startswith("HEAD_MARKER_")
        assert truncated.endswith("_TAIL_MARKER")

    def test_custom_config_limits(self) -> None:
        config = TruncationConfig(
            char_limits={"read_file": 100},
            line_limits={},
        )
        text = "x" * 200
        truncated, full = truncate_output("read_file", text, config=config)
        assert len(full) == 200
        assert "characters removed" in truncated

    def test_unknown_tool_no_truncation(self) -> None:
        text = "x" * 100_000
        truncated, full = truncate_output("unknown_tool", text)
        # No char limit defined for unknown_tool, so no truncation
        assert truncated == text
        assert full == text


class TestLineTruncation:
    """Stage 2 — line-based truncation."""

    def test_shell_line_truncation(self) -> None:
        lines = [f"line {i}" for i in range(1000)]
        text = "\n".join(lines)
        truncated, full = truncate_output("shell", text)
        assert full == text
        assert "lines removed from middle of output" in truncated

    def test_no_line_truncation_when_under_limit(self) -> None:
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        # Shell char limit is 30000, and 50 short lines is well under
        truncated, _ = truncate_output("shell", text)
        assert "lines removed" not in truncated

    def test_both_stages_applied(self) -> None:
        # Create output that triggers both char and line truncation
        config = TruncationConfig(
            char_limits={"shell": 500},
            line_limits={"shell": 10},
        )
        text = "\n".join(f"line-{i}-{'x' * 50}" for i in range(100))
        truncated, full = truncate_output("shell", text, config=config)
        assert full == text
        assert "removed" in truncated


class TestTruncationReturnsTuple:
    """Verify the (truncated, full) return semantics."""

    def test_full_always_original(self) -> None:
        text = "a" * 200_000
        truncated, full = truncate_output("read_file", text)
        assert full == text
        assert truncated != full

    def test_no_truncation_both_equal(self) -> None:
        text = "short"
        truncated, full = truncate_output("read_file", text)
        assert truncated == full == text
