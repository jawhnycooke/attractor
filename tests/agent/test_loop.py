"""Tests for the AgentLoop — agentic tool-call cycle engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor.agent.events import EventEmitter
from attractor.agent.loop import AgentLoop, LoopConfig
from attractor.agent.loop_detection import LoopDetector
from attractor.agent.tools.registry import ToolRegistry, ToolResult
from attractor.agent.truncation import TruncationConfig
from attractor.llm.models import (
    Message,
    Response,
    Role,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


def _make_text_response(text: str) -> Response:
    """Create a Response with only text content (no tool calls)."""
    return Response(
        message=Message(role=Role.ASSISTANT, content=[TextContent(text=text)])
    )


def _make_tool_response(
    tool_name: str, arguments: dict, tool_call_id: str = "call_abc"
) -> Response:
    """Create a Response containing a single tool call."""
    return Response(
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ToolCallContent(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            ],
        )
    )


def _make_profile() -> MagicMock:
    """Create a minimal mock ProviderProfile."""
    profile = MagicMock()
    profile.provider_name = "test"
    profile.tool_definitions = []
    profile.system_prompt_template = "You are a test agent."
    profile.context_window_size = 128_000
    profile.get_tools.return_value = []
    profile.format_system_prompt.return_value = "You are a test agent."
    return profile


def _make_env() -> MagicMock:
    """Create a minimal mock ExecutionEnvironment."""
    env = MagicMock()
    env.working_directory.return_value = "/tmp/test"
    env.platform.return_value = "linux"
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_llm_exception_does_not_corrupt_history(self) -> None:
        """Regression for B4: on LLM error, history should remain in a
        valid state with even length (user/assistant alternation)."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model")

        # Patch build_system_prompt to avoid git calls
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        # Patch the system prompt builder to skip git calls
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # History should be empty — the user message was popped after LLM failure
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_text_only_response_ends_loop(self) -> None:
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model")

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_text_response("Done!"))

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # One user message + one assistant message = 2
        assert len(history) == 2
        assert history[0].role == Role.USER
        assert history[1].role == Role.ASSISTANT
        # LLM called exactly once
        assert llm.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_max_tool_rounds_distinct_from_max_turns(self) -> None:
        """max_tool_rounds should limit independently of max_turns."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        # Set max_tool_rounds to 1, max_turns to 10
        config = LoopConfig(
            model_id="test-model",
            max_turns=10,
            max_tool_rounds=1,
        )

        # First call returns a tool call, but loop should stop at 1 round
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_make_tool_response("read_file", {"path": "x.py"})
        )

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output="file content", full_output="file content")

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # The loop should have done exactly 1 tool round then stopped
        # (1 user + 1 assistant(tool_call) + 1 tool_result)
        # Plus the turn limit is hit at turn=1 because max_tool_rounds=1
        assert llm.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_results_appended_with_correct_ids(self) -> None:
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model", max_turns=2)

        # First call: two tool calls; second call: text-only
        two_tools = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ToolCallContent(
                        tool_call_id="id_1",
                        tool_name="read_file",
                        arguments={"path": "a.py"},
                    ),
                    ToolCallContent(
                        tool_call_id="id_2",
                        tool_name="read_file",
                        arguments={"path": "b.py"},
                    ),
                ],
            )
        )
        text_resp = _make_text_response("All done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[two_tools, text_resp])

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(
                output=f"content of {args.get('path', '')}",
                full_output=f"content of {args.get('path', '')}",
            )

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # Find tool result messages
        tool_results = [m for m in history if m.role == Role.TOOL]
        assert len(tool_results) == 2

        # Verify IDs match
        result_ids = set()
        for msg in tool_results:
            for part in msg.content:
                if isinstance(part, ToolResultContent):
                    result_ids.add(part.tool_call_id)
        assert result_ids == {"id_1", "id_2"}

    @pytest.mark.asyncio
    async def test_steering_drained_before_first_llm_call(self) -> None:
        """Steering queued before submit() must appear before the first LLM call.

        Per spec §2.5: drain_steering() is called BEFORE the first LLM call,
        so steering messages queued while the session is IDLE appear in the
        request messages on the very first LLM invocation.
        """
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model", max_turns=3)

        text_resp = _make_text_response("Done")

        captured_requests: list = []

        async def capturing_complete(request):
            captured_requests.append(request)
            return text_resp

        llm = AsyncMock()
        llm.complete = capturing_complete

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        # Queue steering BEFORE running (simulates steer() called while IDLE)
        loop.queue_steering("Focus on tests only")

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # The steering message must appear in history BEFORE the first LLM call.
        # Since we captured the request, the messages list sent to the LLM
        # should contain the steering message (as a user message) in the
        # first call's messages.
        assert len(captured_requests) == 1
        first_request = captured_requests[0]
        first_request_messages = first_request.messages

        # History order should be: user("hello"), user("Focus on tests only")
        # Both should be in the first request's messages.
        user_texts = [
            m.text() for m in first_request_messages if m.role == Role.USER
        ]
        assert "Focus on tests only" in user_texts

        # Also verify the steering message appears in history before the
        # assistant response
        assert history[0].role == Role.USER  # "hello"
        assert history[1].role == Role.USER  # steering: "Focus on tests only"
        assert "Focus on tests only" in history[1].text()
        assert history[2].role == Role.ASSISTANT  # "Done"

    @pytest.mark.asyncio
    async def test_steering_messages_injected_between_rounds(self) -> None:
        """Tests B2: steering messages are injected as user messages."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model", max_turns=3)

        tool_resp = _make_tool_response("read_file", {"path": "a.py"})
        text_resp = _make_text_response("Done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[tool_resp, text_resp])

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output="content", full_output="content")

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        # Queue a steering message before running
        loop.queue_steering("Try a different approach")

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # Find the steering message in history
        user_msgs = [
            m
            for m in history
            if m.role == Role.USER and "different approach" in m.text()
        ]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_loop_detection_injects_warning(self) -> None:
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(
            model_id="test-model",
            enable_loop_detection=True,
            max_turns=10,
        )

        # Make the LLM always return the same tool call
        _make_tool_response("read_file", {"path": "same.py"}, tool_call_id="call_loop")
        # After enough repetitions, the loop detector should fire.
        # Then on the next call, return text to end.
        call_count = 0

        async def fake_complete(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return _make_tool_response(
                    "read_file",
                    {"path": "same.py"},
                    tool_call_id=f"call_{call_count}",
                )
            return _make_text_response("giving up")

        llm = AsyncMock()
        llm.complete = fake_complete

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output="content", full_output="content")

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # There should be a warning message injected about loop detection
        warning_msgs = [
            m for m in history if m.role == Role.USER and "SYSTEM WARNING" in m.text()
        ]
        assert len(warning_msgs) >= 1

    @pytest.mark.asyncio
    async def test_truncation_applied_to_long_output(self) -> None:
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        trunc_config = TruncationConfig(
            char_limits={"read_file": 100},
            line_limits={},
        )
        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            truncation_config=trunc_config,
        )

        tool_resp = _make_tool_response("read_file", {"path": "big.py"})
        text_resp = _make_text_response("Done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[tool_resp, text_resp])

        big_output = "x" * 10_000

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output=big_output, full_output=big_output)

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # The tool result in history should be truncated (contain the marker)
        tool_result_msgs = [m for m in history if m.role == Role.TOOL]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0].content[0].content
        assert "truncated" in result_content.lower()
        assert len(result_content) < len(big_output)

    @pytest.mark.asyncio
    async def test_max_command_timeout_clamped(self) -> None:
        """Shell tool timeout_ms should be clamped to max_command_timeout_ms."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        # Set a low max_command_timeout_ms ceiling
        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            max_command_timeout_ms=5_000,
        )

        # LLM requests shell with a timeout exceeding the max
        tool_resp = _make_tool_response(
            "shell",
            {"command": "sleep 60", "timeout_ms": 999_999},
        )
        text_resp = _make_text_response("Done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[tool_resp, text_resp])

        captured_args: list[dict] = []

        async def fake_dispatch(name, args, env_arg):
            captured_args.append(dict(args))
            return ToolResult(output="ok", full_output="ok")

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("run it", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # The timeout_ms in the dispatched args should be clamped to 5000
        assert len(captured_args) == 1
        assert captured_args[0]["timeout_ms"] == 5_000

    @pytest.mark.asyncio
    async def test_loop_detection_uses_configured_window(self) -> None:
        """Loop detection should use the configured window_size from config."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        # Use a small window so loop triggers faster
        config = LoopConfig(
            model_id="test-model",
            enable_loop_detection=True,
            loop_detection_window=3,
            max_turns=10,
        )

        call_count = 0

        async def fake_complete(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return _make_tool_response(
                    "read_file",
                    {"path": "same.py"},
                    tool_call_id=f"call_{call_count}",
                )
            return _make_text_response("giving up")

        llm = AsyncMock()
        llm.complete = fake_complete

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output="content", full_output="content")

        registry.dispatch = fake_dispatch

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
        )

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # With window_size=3 and repeating single calls, loop should be detected
        warning_msgs = [
            m for m in history if m.role == Role.USER and "SYSTEM WARNING" in m.text()
        ]
        assert len(warning_msgs) >= 1
