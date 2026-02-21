"""Tests for the loop detection module."""

from attractor.agent.loop_detection import LoopDetector


class TestLoopDetector:
    """LoopDetector pattern detection tests."""

    def test_no_loop_with_few_calls(self) -> None:
        detector = LoopDetector()
        detector.record_call("read_file", "abc123")
        detector.record_call("edit_file", "def456")
        assert detector.check_for_loops() is None

    def test_single_call_loop(self) -> None:
        """Same tool+args repeated 3+ times should be detected."""
        detector = LoopDetector()
        for _ in range(5):
            detector.record_call("read_file", "same_hash")
        warning = detector.check_for_loops()
        assert warning is not None
        assert "1-call pattern" in warning

    def test_two_call_cycle(self) -> None:
        """Alternating between two tool calls should be detected."""
        detector = LoopDetector()
        for _ in range(5):
            detector.record_call("read_file", "hash_a")
            detector.record_call("edit_file", "hash_b")
        warning = detector.check_for_loops()
        assert warning is not None
        assert "2-call pattern" in warning

    def test_three_call_cycle(self) -> None:
        """A repeating 3-call sequence should be detected."""
        detector = LoopDetector()
        for _ in range(4):
            detector.record_call("read_file", "a")
            detector.record_call("edit_file", "b")
            detector.record_call("shell", "c")
        warning = detector.check_for_loops()
        assert warning is not None
        assert "3-call pattern" in warning

    def test_no_false_positive_for_varied_calls(self) -> None:
        """Diverse call patterns should not trigger detection."""
        detector = LoopDetector()
        for i in range(10):
            detector.record_call(f"tool_{i}", f"hash_{i}")
        assert detector.check_for_loops() is None

    def test_reset_clears_history(self) -> None:
        detector = LoopDetector()
        for _ in range(5):
            detector.record_call("read_file", "same")
        assert detector.check_for_loops() is not None
        detector.reset()
        assert detector.check_for_loops() is None

    def test_window_size_respected(self) -> None:
        """Loops outside the window should not be detected."""
        detector = LoopDetector()
        # Fill with loops, then different calls
        for _ in range(5):
            detector.record_call("read_file", "same")
        for i in range(15):
            detector.record_call(f"tool_{i}", f"hash_{i}")
        # The loop was 15+ calls ago, outside default window of 10
        assert detector.check_for_loops() is None

    def test_window_boundary_just_inside(self) -> None:
        """A loop pattern at exactly the window edge should still be detected."""
        detector = LoopDetector()
        # Add some non-looping calls first to push the loop to the window edge
        for i in range(7):
            detector.record_call(f"setup_{i}", f"s_{i}")
        # Now add a 1-call loop (3 repetitions = 3 calls)
        # With window=10, these 3 calls are within the last 10
        for _ in range(3):
            detector.record_call("read_file", "same")
        assert detector.check_for_loops(window_size=10) is not None

    def test_window_boundary_just_outside(self) -> None:
        """A loop pushed just outside the window should not be detected."""
        detector = LoopDetector()
        # Add a 1-call loop
        for _ in range(3):
            detector.record_call("read_file", "same")
        # Now add exactly enough unique calls to push the loop out of window
        for i in range(10):
            detector.record_call(f"unique_{i}", f"u_{i}")
        # Window of 10 only sees the 10 unique calls, not the loop
        assert detector.check_for_loops(window_size=10) is None
