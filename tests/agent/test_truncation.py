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
        """Char truncation runs first, then line truncation on the result.

        With char-first ordering:
          1. Char truncation: 10000+ chars → ~1000 chars (~18 lines)
          2. Line truncation: ~18 lines → 10 lines

        If lines ran first (hypothetical reverse order):
          1. Line truncation: 200 lines → 10 lines (~550 chars)
          2. Char truncation: ~550 chars → no truncation (under 1000 limit)

        So char-first produces a shorter result than line-first would.
        """
        config = TruncationConfig(
            char_limits={"shell": 1000},
            line_limits={"shell": 10},
        )
        # 200 lines of ~54 chars each ≈ 10,800 chars
        text = "\n".join(f"line-{i}-{'x' * 50}" for i in range(200))
        truncated, full = truncate_output("shell", text, config=config)
        assert full == text

        # Simulate line-only truncation to prove both stages contribute
        line_only_config = TruncationConfig(
            char_limits={},  # no char truncation
            line_limits={"shell": 10},
        )
        line_only_result, _ = truncate_output("shell", text, config=line_only_config)

        # With both stages, the result should be shorter than line-only
        # because char truncation runs first and introduces a marker line,
        # making subsequent line truncation operate on a different structure
        assert len(truncated) < len(line_only_result)
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
