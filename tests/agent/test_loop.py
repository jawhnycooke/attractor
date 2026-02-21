"""Tests for the AgentLoop — agentic tool-call cycle engine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor.agent.events import AgentEvent, AgentEventType, EventEmitter
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
    ToolDefinition,
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

        # Warning is injected on each turn after the pattern is first detected
        # (calls 3, 4, and 5 all trigger warnings)
        warning_msgs = [
            m for m in history if m.role == Role.USER and "SYSTEM WARNING" in m.text()
        ]
        assert len(warning_msgs) == 3

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
        # Warning fires on each turn after pattern first detected (calls 3, 4, 5)
        warning_msgs = [
            m for m in history if m.role == Role.USER and "SYSTEM WARNING" in m.text()
        ]
        assert len(warning_msgs) == 3


# ---------------------------------------------------------------------------
# ISSUE #1: Enhanced tool argument validation
# ---------------------------------------------------------------------------


class TestToolArgumentValidation:
    """Tests for extended JSON schema validation in _validate_tool_arguments."""

    def _make_loop_with_tool(
        self, tool_name: str, schema: dict
    ) -> AgentLoop:
        """Build an AgentLoop with a single registered tool definition."""
        registry = ToolRegistry()
        definition = ToolDefinition(
            name=tool_name,
            description="test tool",
            parameters=schema,
        )
        # Register with a dummy handler
        async def noop(arguments, environment):
            return ToolResult(output="ok")

        registry.register(tool_name, noop, definition)

        return AgentLoop(
            profile=_make_profile(),
            environment=_make_env(),
            registry=registry,
            llm_client=AsyncMock(),
            emitter=EventEmitter(),
            config=LoopConfig(model_id="test-model"),
            loop_detector=LoopDetector(),
        )

    def test_enum_valid(self) -> None:
        schema = {
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"mode": "fast"})
        assert result is None

    def test_enum_invalid(self) -> None:
        schema = {
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"mode": "turbo"}
        )
        assert result is not None
        assert "must be one of" in result
        assert "turbo" in result

    def test_pattern_valid(self) -> None:
        schema = {
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": r"^[^@]+@[^@]+\.[^@]+$",
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"email": "a@b.com"}
        )
        assert result is None

    def test_pattern_invalid(self) -> None:
        schema = {
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": r"^[^@]+@[^@]+\.[^@]+$",
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"email": "not-an-email"}
        )
        assert result is not None
        assert "does not match pattern" in result

    def test_min_length_valid(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string", "minLength": 3},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"name": "abc"}
        )
        assert result is None

    def test_min_length_invalid(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string", "minLength": 3},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"name": "ab"})
        assert result is not None
        assert "below minimum" in result
        assert "3" in result

    def test_max_length_valid(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string", "maxLength": 5},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"name": "hello"}
        )
        assert result is None

    def test_max_length_invalid(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string", "maxLength": 5},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"name": "toolong"}
        )
        assert result is not None
        assert "exceeds maximum" in result
        assert "5" in result

    def test_minimum_valid(self) -> None:
        schema = {
            "properties": {
                "count": {"type": "integer", "minimum": 1},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"count": 1})
        assert result is None

    def test_minimum_invalid(self) -> None:
        schema = {
            "properties": {
                "count": {"type": "integer", "minimum": 1},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"count": 0})
        assert result is not None
        assert "below minimum" in result

    def test_maximum_valid(self) -> None:
        schema = {
            "properties": {
                "count": {"type": "integer", "maximum": 100},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"count": 100})
        assert result is None

    def test_maximum_invalid(self) -> None:
        schema = {
            "properties": {
                "count": {"type": "integer", "maximum": 100},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {"count": 101})
        assert result is not None
        assert "exceeds maximum" in result

    def test_array_items_valid(self) -> None:
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"tags": ["a", "b", "c"]}
        )
        assert result is None

    def test_array_items_invalid(self) -> None:
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"tags": ["a", 123, "c"]}
        )
        assert result is not None
        assert "tags[1]" in result
        assert "string" in result

    def test_array_empty_passes(self) -> None:
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"tags": []}
        )
        assert result is None

    def test_nested_object_valid(self) -> None:
        schema = {
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                        "retries": {"type": "integer"},
                    },
                    "required": ["timeout"],
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"config": {"timeout": 30, "retries": 3}}
        )
        assert result is None

    def test_nested_object_missing_required(self) -> None:
        schema = {
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                    },
                    "required": ["timeout"],
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"config": {"retries": 3}}
        )
        assert result is not None
        assert "Missing required field 'timeout'" in result

    def test_nested_object_wrong_type(self) -> None:
        schema = {
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                    },
                },
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments(
            "my_tool", {"config": {"timeout": "not_int"}}
        )
        assert result is not None
        assert "config.timeout" in result
        assert "integer" in result

    def test_unknown_tool_returns_none(self) -> None:
        loop = self._make_loop_with_tool("other_tool", {})
        result = loop._validate_tool_arguments(
            "unknown_tool", {"foo": "bar"}
        )
        assert result is None

    def test_empty_schema_passes(self) -> None:
        loop = self._make_loop_with_tool("my_tool", {})
        result = loop._validate_tool_arguments(
            "my_tool", {"anything": "goes"}
        )
        assert result is None

    def test_missing_required_still_works(self) -> None:
        schema = {
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        result = loop._validate_tool_arguments("my_tool", {})
        assert result is not None
        assert "Missing required argument 'path'" in result

    def test_numeric_constraints_ignore_booleans(self) -> None:
        """Booleans are a subclass of int in Python; numeric constraints
        should not apply to them."""
        schema = {
            "properties": {
                "flag": {"type": "boolean", "minimum": 5},
            },
        }
        loop = self._make_loop_with_tool("my_tool", schema)
        # True is 1, which is < 5, but this should pass because it's a bool
        result = loop._validate_tool_arguments("my_tool", {"flag": True})
        assert result is None


# ---------------------------------------------------------------------------
# ISSUE #4: TOOL_CALL_END event carries full output
# ---------------------------------------------------------------------------


class TestToolCallEndEventOutput:
    """Tests that TOOL_CALL_END emits full output, not truncated."""

    @pytest.mark.asyncio
    async def test_tool_call_end_has_full_output(self) -> None:
        """TOOL_CALL_END event 'output' field must be untruncated."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        trunc_config = TruncationConfig(
            char_limits={"read_file": 50},
            line_limits={},
        )
        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            truncation_config=trunc_config,
        )

        tool_resp = _make_tool_response(
            "read_file", {"path": "big.py"}, tool_call_id="tc_1"
        )
        text_resp = _make_text_response("Done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[tool_resp, text_resp])

        full_output = "a" * 5_000

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output=full_output, full_output=full_output)

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

        collected_events: list[AgentEvent] = []
        original_emit = emitter.emit

        def capturing_emit(event: AgentEvent) -> None:
            collected_events.append(event)
            original_emit(event)

        emitter.emit = capturing_emit

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original_prompt = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original_prompt

        emitter.close()

        # Find the TOOL_CALL_END event
        end_events = [
            e
            for e in collected_events
            if e.type == AgentEventType.TOOL_CALL_END
        ]
        assert len(end_events) == 1
        end_data = end_events[0].data

        # 'output' must be the FULL untruncated output
        assert end_data["output"] == full_output
        # 'truncated_output' must be shorter than the full output
        assert len(end_data["truncated_output"]) < len(full_output)
        # The old 'full_output' key must NOT be present
        assert "full_output" not in end_data

    @pytest.mark.asyncio
    async def test_tool_call_end_no_truncation_same_output(self) -> None:
        """When output is small enough, output and truncated_output match."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        config = LoopConfig(model_id="test-model", max_turns=2)

        tool_resp = _make_tool_response(
            "read_file", {"path": "small.py"}, tool_call_id="tc_1"
        )
        text_resp = _make_text_response("Done")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=[tool_resp, text_resp])

        small_output = "hello world"

        async def fake_dispatch(name, args, env_arg):
            return ToolResult(output=small_output, full_output=small_output)

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

        collected_events: list[AgentEvent] = []
        original_emit = emitter.emit

        def capturing_emit(event: AgentEvent) -> None:
            collected_events.append(event)
            original_emit(event)

        emitter.emit = capturing_emit

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original_prompt = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original_prompt

        emitter.close()

        end_events = [
            e
            for e in collected_events
            if e.type == AgentEventType.TOOL_CALL_END
        ]
        assert len(end_events) == 1
        end_data = end_events[0].data

        # Both should be the same when no truncation needed
        assert end_data["output"] == small_output
        assert end_data["truncated_output"] == small_output


# ---------------------------------------------------------------------------
# ISSUE #6: Shell tool default timeout from config
# ---------------------------------------------------------------------------


class TestShellDefaultTimeout:
    """Tests that shell tool uses config.default_command_timeout_ms when
    no explicit timeout_ms is provided by the LLM."""

    @pytest.mark.asyncio
    async def test_shell_no_timeout_uses_config_default(self) -> None:
        """Shell without timeout_ms should get default_command_timeout_ms."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            default_command_timeout_ms=120_000,
        )

        # Shell call with NO timeout_ms
        tool_resp = _make_tool_response(
            "shell", {"command": "ls -la"}
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
            await loop.run("run ls", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        assert len(captured_args) == 1
        assert captured_args[0]["timeout_ms"] == 120_000

    @pytest.mark.asyncio
    async def test_shell_explicit_timeout_not_overridden(self) -> None:
        """Shell with explicit timeout_ms below ceiling should keep it."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            default_command_timeout_ms=120_000,
            max_command_timeout_ms=600_000,
        )

        # Shell call with explicit timeout under the ceiling
        tool_resp = _make_tool_response(
            "shell", {"command": "echo hi", "timeout_ms": 5_000}
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
            await loop.run("run echo", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        assert len(captured_args) == 1
        assert captured_args[0]["timeout_ms"] == 5_000

    @pytest.mark.asyncio
    async def test_non_shell_tool_no_timeout_added(self) -> None:
        """Non-shell tools should NOT get a timeout_ms injected."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()

        config = LoopConfig(
            model_id="test-model",
            max_turns=2,
            default_command_timeout_ms=120_000,
        )

        tool_resp = _make_tool_response(
            "read_file", {"path": "foo.py"}
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
            await loop.run("read file", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        assert len(captured_args) == 1
        assert "timeout_ms" not in captured_args[0]


# ---------------------------------------------------------------------------
# ISSUE #2: Abort signal support
# ---------------------------------------------------------------------------


class TestAbortSignal:
    """Tests for abort_event parameter on AgentLoop."""

    @pytest.mark.asyncio
    async def test_abort_before_first_llm_call(self) -> None:
        """If abort is set before the loop starts, it exits immediately."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model")

        abort_event = asyncio.Event()
        abort_event.set()  # Pre-set abort

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_text_response("Done"))

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
            abort_event=abort_event,
        )

        collected_events: list[AgentEvent] = []
        original_emit = emitter.emit

        def capturing_emit(event: AgentEvent) -> None:
            collected_events.append(event)
            original_emit(event)

        emitter.emit = capturing_emit

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # LLM should never have been called
        assert llm.complete.call_count == 0

        # Error event with abort phase emitted
        abort_events = [
            e
            for e in collected_events
            if e.type == AgentEventType.ERROR
            and e.data.get("phase") == "abort"
        ]
        assert len(abort_events) == 1
        assert abort_events[0].data["error"] == "Aborted"

    @pytest.mark.asyncio
    async def test_abort_after_tool_execution(self) -> None:
        """Abort set during tool execution stops the loop after that round."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model", max_turns=10)

        abort_event = asyncio.Event()

        tool_resp = _make_tool_response(
            "read_file", {"path": "a.py"}, tool_call_id="tc_1"
        )

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=tool_resp)

        async def fake_dispatch(name, args, env_arg):
            # Set abort during tool execution
            abort_event.set()
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
            abort_event=abort_event,
        )

        collected_events: list[AgentEvent] = []
        original_emit = emitter.emit

        def capturing_emit(event: AgentEvent) -> None:
            collected_events.append(event)
            original_emit(event)

        emitter.emit = capturing_emit

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # LLM called exactly once (then abort stops the loop)
        assert llm.complete.call_count == 1

        # Error event with abort phase emitted
        abort_events = [
            e
            for e in collected_events
            if e.type == AgentEventType.ERROR
            and e.data.get("phase") == "abort"
        ]
        assert len(abort_events) == 1
        assert abort_events[0].data["error"] == "Aborted"

    @pytest.mark.asyncio
    async def test_no_abort_event_runs_normally(self) -> None:
        """With no abort_event (None), the loop runs to completion."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model")

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_text_response("Done"))

        loop = AgentLoop(
            profile=profile,
            environment=env,
            registry=registry,
            llm_client=llm,
            emitter=emitter,
            config=config,
            loop_detector=detector,
            abort_event=None,
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

        # Normal completion — LLM called once, text response
        assert llm.complete.call_count == 1
        assert len(history) == 2
        assert history[1].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_abort_event_not_set_does_not_stop(self) -> None:
        """An abort_event that is NOT set should not stop the loop."""
        profile = _make_profile()
        env = _make_env()
        registry = ToolRegistry()
        emitter = EventEmitter()
        detector = LoopDetector()
        config = LoopConfig(model_id="test-model", max_turns=5)

        abort_event = asyncio.Event()  # Not set

        tool_resp = _make_tool_response(
            "read_file", {"path": "a.py"}, tool_call_id="tc_1"
        )
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
            abort_event=abort_event,
        )

        collected_events: list[AgentEvent] = []
        original_emit = emitter.emit

        def capturing_emit(event: AgentEvent) -> None:
            collected_events.append(event)
            original_emit(event)

        emitter.emit = capturing_emit

        history: list[Message] = []
        import attractor.agent.loop as loop_mod

        original = loop_mod.build_system_prompt
        loop_mod.build_system_prompt = AsyncMock(return_value="system prompt")
        try:
            await loop.run("hello", history)
        finally:
            loop_mod.build_system_prompt = original

        emitter.close()

        # Should complete normally: 1 tool round + 1 text response = 2 calls
        assert llm.complete.call_count == 2

        # No abort error events
        abort_events = [
            e
            for e in collected_events
            if e.type == AgentEventType.ERROR
            and e.data.get("phase") == "abort"
        ]
        assert len(abort_events) == 0
