"""Loop detection for the agentic coding loop.

Detects repeating patterns of tool calls (cycles of length 1, 2, or 3)
within a sliding window to prevent the agent from getting stuck.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _hash_call(tool_name: str, arguments_hash: str) -> str:
    """Create a compact fingerprint for a single tool call."""
    raw = f"{tool_name}:{arguments_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class LoopDetector:
    """Tracks tool call history and detects repeating patterns."""

    _history: list[str] = field(default_factory=list)

    def record_call(self, tool_name: str, arguments_hash: str) -> None:
        """Record a tool call fingerprint."""
        self._history.append(_hash_call(tool_name, arguments_hash))

    def check_for_loops(self, window_size: int = 10) -> str | None:
        """Check the last *window_size* calls for repeating cycles.

        Looks for repeating patterns of length 1, 2, and 3 within the
        window. A pattern is considered a loop if it repeats at least
        3 consecutive times.

        Returns:
            A warning message string if a loop is detected, or ``None``.
        """
        window = self._history[-window_size:]
        if len(window) < 3:
            return None

        # Check cycle lengths 1, 2, 3
        for cycle_len in (1, 2, 3):
            required = cycle_len * 3  # need 3 repetitions
            if len(window) < required:
                continue

            tail = window[-required:]
            pattern = tail[:cycle_len]
            is_loop = True
            for i in range(cycle_len, required):
                if tail[i] != pattern[i % cycle_len]:
                    is_loop = False
                    break

            if is_loop:
                return (
                    f"Loop detected: the same {cycle_len}-call pattern "
                    f"has repeated {required // cycle_len} times in the "
                    f"last {len(window)} tool calls. Consider a different "
                    f"approach."
                )

        return None

    def reset(self) -> None:
        """Clear recorded history."""
        self._history.clear()
