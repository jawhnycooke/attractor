"""Agent subsystem integration smoke tests (spec §9.13).

Requires a real ANTHROPIC_API_KEY.  Run with:

    pytest -m smoke tests/agent/test_smoke_agent.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.events import AgentEventType
from attractor.agent.profiles import AnthropicProfile
from attractor.agent.session import Session, SessionConfig
from attractor.llm.client import LLMClient

SMOKE_MODEL = "claude-sonnet-4-6"

pytestmark = pytest.mark.smoke


async def _make_session(
    tmp_path: Path,
    *,
    max_turns: int = 10,
) -> Session:
    """Create a Session wired to a real LLM in a temp directory."""
    env = LocalExecutionEnvironment(working_dir=str(tmp_path))
    await env.initialize()
    return Session(
        profile=AnthropicProfile(),
        environment=env,
        config=SessionConfig(model_id=SMOKE_MODEL, max_turns=max_turns),
        llm_client=LLMClient.from_env(),
    )


async def _collect_events(session: Session, prompt: str) -> list:
    """Submit a prompt and collect all events."""
    events = []
    async for event in session.submit(prompt):
        events.append(event)
    return events


class TestAgentSmoke:
    """Smoke tests for the coding agent against a real Anthropic endpoint."""

    # §9.13 step 1 — file creation
    async def test_file_creation(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        session = await _make_session(tmp_path)
        await _collect_events(
            session,
            "Create a file called hello.py that contains a print statement "
            "printing 'Hello World'. Use the write_file tool.",
        )

        hello = tmp_path / "hello.py"
        assert hello.exists(), "Expected hello.py to be created"
        content = hello.read_text()
        assert "print" in content.lower(), "Expected a print statement in hello.py"
        assert "hello" in content.lower(), "Expected 'hello' in the file content"

    # §9.13 step 2 — read and edit an existing file
    async def test_read_and_edit(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        # Pre-create a file the agent can read and edit
        target = tmp_path / "greet.py"
        target.write_text('print("Hello")\n')

        session = await _make_session(tmp_path)
        await _collect_events(
            session,
            "Read the file greet.py, then add a second line that prints "
            "'Goodbye'. Keep the existing Hello line.",
        )

        content = target.read_text()
        assert "Hello" in content, "Original Hello line should still be present"
        assert "Goodbye" in content or "goodbye" in content.lower(), (
            "Expected a Goodbye line to be added"
        )

    # §9.13 step 3 — shell execution
    async def test_shell_execution(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        # Pre-create a file to run
        script = tmp_path / "hi.py"
        script.write_text('print("hi from script")\n')

        session = await _make_session(tmp_path)
        events = await _collect_events(
            session,
            "Run the command 'python hi.py' using the shell tool and tell me "
            "the output.",
        )

        tool_starts = [
            e for e in events if e.type == AgentEventType.TOOL_CALL_START
        ]
        tool_names = [e.data.get("tool_name", "") for e in tool_starts]
        assert "shell" in tool_names, (
            f"Expected a shell tool call; got tool calls: {tool_names}"
        )

    # §9.13 step 4 — truncation of large tool output
    async def test_truncation(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        # Write a 100 KB file — large enough to trigger truncation
        big_file = tmp_path / "big.txt"
        big_file.write_text("x" * 100_000 + "\n")

        session = await _make_session(tmp_path, max_turns=5)
        events = await _collect_events(
            session,
            "Read the file big.txt and tell me how many characters it contains.",
        )

        # Find the TOOL_CALL_END event for the read_file tool.
        # Event data keys: "output" = full content, "truncated_output" = LLM-facing.
        read_ends = [
            e
            for e in events
            if e.type == AgentEventType.TOOL_CALL_END
            and e.data.get("tool_name") == "read_file"
        ]
        assert len(read_ends) > 0, (
            "Expected at least one read_file TOOL_CALL_END event"
        )
        full_output = read_ends[0].data.get("output", "")
        truncated_output = read_ends[0].data.get("truncated_output", "")
        assert len(full_output) > 0, "Expected non-empty output"
        assert truncated_output, (
            "Expected truncated_output to be present for 100KB file"
        )
        assert len(truncated_output) < len(full_output), (
            "Expected truncated_output to be shorter than full output"
        )

    # §9.13 step 5 — steering injection
    async def test_steering(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        # Create some files so the agent has something to explore
        for name in ("a.py", "b.py", "c.py"):
            (tmp_path / name).write_text(f'# {name}\nprint("{name}")\n')

        session = await _make_session(tmp_path, max_turns=15)

        steered = False
        events = []
        async for event in session.submit(
            "List all Python files in the current directory. "
            "Read each one and summarize its contents."
        ):
            events.append(event)
            # After the first tool call completes, inject steering
            if (
                not steered
                and event.type == AgentEventType.TOOL_CALL_END
            ):
                session.steer(
                    "Stop reading files. Just reply with 'Done' immediately."
                )
                steered = True

        assert steered, (
            "Expected at least one TOOL_CALL_END event to trigger steering"
        )
        steering_events = [
            e
            for e in events
            if e.type == AgentEventType.STEERING_INJECTED
        ]
        assert len(steering_events) > 0, (
            "Expected STEERING_INJECTED event after calling steer()"
        )
