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
        assert "characters were removed from the middle" in truncated

    def test_head_tail_warning_matches_spec(self) -> None:
        """Warning text must match spec §5.1 exactly."""
        text = "x" * 100_000
        truncated, _ = truncate_output("read_file", text)
        assert "characters were removed from the middle." in truncated
        assert "The full output is available in the event stream." in truncated
        assert "re-run the tool with more targeted parameters." in truncated

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
        assert "characters were removed" in truncated

    def test_unknown_tool_no_truncation(self) -> None:
        text = "x" * 100_000
        truncated, full = truncate_output("unknown_tool", text)
        # No char limit defined for unknown_tool, so no truncation
        assert truncated == text
        assert full == text

    def test_spawn_agent_in_char_limits(self) -> None:
        """spawn_agent has a 20,000 char limit per spec §5.2."""
        cfg = TruncationConfig()
        assert "spawn_agent" in cfg.char_limits
        assert cfg.char_limits["spawn_agent"] == 20_000

    def test_spawn_agent_truncated(self) -> None:
        """spawn_agent output exceeding 20k chars is truncated with head_tail mode."""
        text = "HEAD_" + "x" * 50_000 + "_TAIL"
        truncated, full = truncate_output("spawn_agent", text)
        assert full == text
        assert len(truncated) < len(full)
        assert truncated.startswith("HEAD_")
        assert truncated.endswith("_TAIL")

    def test_spawn_agent_uses_head_tail_mode(self) -> None:
        """spawn_agent uses head_tail mode (not tail)."""
        cfg = TruncationConfig()
        assert cfg.truncation_modes.get("spawn_agent") == "head_tail"


class TestTailTruncation:
    """Stage 1 — tail-only character truncation mode."""

    def test_short_output_unchanged(self) -> None:
        """Output under the limit is returned as-is even in tail mode."""
        truncated, full = truncate_output("grep", "short output")
        assert truncated == "short output"
        assert full == "short output"

    def test_tail_mode_keeps_end(self) -> None:
        """Tail mode discards the beginning and keeps the end."""
        tail_content = "TAIL_MARKER_" + "z" * 1000
        text = "x" * 50_000 + tail_content
        truncated, full = truncate_output("grep", text)
        assert full == text
        assert truncated.endswith(tail_content)
        assert not truncated.startswith("x")

    def test_tail_mode_warning_text(self) -> None:
        """Tail mode prepends a warning about removed beginning."""
        text = "x" * 50_000
        truncated, _ = truncate_output("grep", text)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")
        assert "characters were removed" in truncated
        assert "event stream" in truncated

    def test_tail_mode_removes_correct_count(self) -> None:
        """The warning reports the exact number of removed characters."""
        config = TruncationConfig(
            char_limits={"grep": 100},
            truncation_modes={"grep": "tail"},
        )
        text = "x" * 250
        truncated, _ = truncate_output("grep", text, config=config)
        # 250 - 100 = 150 chars removed
        assert "150 characters were removed" in truncated

    def test_tail_mode_applied_to_glob(self) -> None:
        """Glob tool uses tail mode by default."""
        text = "x" * 50_000
        truncated, _ = truncate_output("glob", text)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")

    def test_tail_mode_applied_to_edit_file(self) -> None:
        """edit_file tool uses tail mode by default."""
        text = "x" * 20_000
        truncated, _ = truncate_output("edit_file", text)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")

    def test_tail_mode_applied_to_apply_patch(self) -> None:
        """apply_patch tool uses tail mode by default."""
        text = "x" * 20_000
        truncated, _ = truncate_output("apply_patch", text)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")

    def test_tail_mode_applied_to_write_file(self) -> None:
        """write_file tool uses tail mode by default."""
        text = "x" * 5_000
        truncated, _ = truncate_output("write_file", text)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")

    def test_head_tail_mode_still_works_for_read_file(self) -> None:
        """read_file still uses head_tail mode (not tail)."""
        text = "HEAD_" + "x" * 100_000 + "_TAIL"
        truncated, _ = truncate_output("read_file", text)
        assert truncated.startswith("HEAD_")
        assert truncated.endswith("_TAIL")

    def test_head_tail_mode_still_works_for_shell(self) -> None:
        """shell still uses head_tail mode (not tail)."""
        config = TruncationConfig(
            char_limits={"shell": 200},
            truncation_modes={"shell": "head_tail"},
            line_limits={},
        )
        text = "HEAD_" + "x" * 1000 + "_TAIL"
        truncated, _ = truncate_output("shell", text, config=config)
        assert truncated.startswith("HEAD_")
        assert truncated.endswith("_TAIL")

    def test_custom_mode_override(self) -> None:
        """Custom config can override a tool's truncation mode."""
        config = TruncationConfig(
            char_limits={"read_file": 200},
            truncation_modes={"read_file": "tail"},
            line_limits={},
        )
        text = "x" * 1000
        truncated, _ = truncate_output("read_file", text, config=config)
        assert truncated.startswith("[WARNING: Tool output was truncated. First ")

    def test_default_mode_is_head_tail(self) -> None:
        """Unknown tools default to head_tail mode."""
        config = TruncationConfig(
            char_limits={"custom_tool": 100},
            line_limits={},
        )
        text = "HEAD_" + "x" * 500 + "_TAIL"
        truncated, _ = truncate_output("custom_tool", text, config=config)
        assert truncated.startswith("HEAD_")
        assert truncated.endswith("_TAIL")


class TestLineTruncation:
    """Stage 2 — line-based truncation."""

    def test_shell_line_truncation(self) -> None:
        lines = [f"line {i}" for i in range(1000)]
        text = "\n".join(lines)
        truncated, full = truncate_output("shell", text)
        assert full == text
        assert "lines omitted" in truncated

    def test_line_marker_matches_spec(self) -> None:
        """Line truncation marker must be `[... N lines omitted ...]` per spec §5.3."""
        config = TruncationConfig(
            char_limits={},
            line_limits={"shell": 10},
        )
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        truncated, _ = truncate_output("shell", text, config=config)
        assert "[... 90 lines omitted ...]" in truncated

    def test_no_line_truncation_when_under_limit(self) -> None:
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        # Shell char limit is 30000, and 50 short lines is well under
        truncated, _ = truncate_output("shell", text)
        assert "lines omitted" not in truncated

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
        assert "lines omitted" in truncated


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
