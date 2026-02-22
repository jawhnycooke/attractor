"""Tests for Session lifecycle with a mocked LLM client."""

import uuid

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.events import AgentEvent, AgentEventType
from attractor.agent.profiles.anthropic_profile import AnthropicProfile
from attractor.agent.session import Session, SessionConfig, SessionState
from attractor.llm.models import (
    FinishReason,
    Message,
    ReasoningEffort,
    Request,
    Response,
    Role,
    ToolCallContent,
)


class MockLLMClient:
    """Mock LLM client that returns canned responses."""

    def __init__(self, responses: list[Response] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def complete(self, request: Request) -> Response:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        # Default: return a simple text response
        return Response(
            message=Message.assistant("Done."),
            finish_reason=FinishReason.STOP,
        )


def _text_response(text: str) -> Response:
    return Response(
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
    )


def _tool_call_response(tool_name: str, arguments: dict) -> Response:
    msg = Message(
        role=Role.ASSISTANT,
        content=[
            ToolCallContent(
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=f"call_{tool_name}",
            )
        ],
    )
    return Response(message=msg, finish_reason=FinishReason.TOOL_USE)


@pytest.fixture
async def env(tmp_path):
    environment = LocalExecutionEnvironment(working_dir=str(tmp_path))
    await environment.initialize()
    yield environment
    await environment.cleanup()


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, env) -> None:
        """Session should handle a single text response and emit correct events."""
        client = MockLLMClient([_text_response("Hello!")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events = []
        async for event in session.submit("Hi"):
            events.append(event)

        types = [e.type for e in events]
        assert AgentEventType.SESSION_START in types
        assert AgentEventType.USER_INPUT in types
        assert AgentEventType.ASSISTANT_TEXT_START in types
        assert AgentEventType.ASSISTANT_TEXT_DELTA in types
        assert AgentEventType.ASSISTANT_TEXT_END in types
        assert AgentEventType.SESSION_END in types

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self, env, tmp_path) -> None:
        """Session should execute tool calls and loop back to LLM."""
        (tmp_path / "test.txt").write_text("file content\n")

        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "test.txt"}),
                _text_response("The file contains: file content"),
            ]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events = []
        async for event in session.submit("Read test.txt"):
            events.append(event)

        types = [e.type for e in events]
        assert AgentEventType.TOOL_CALL_START in types
        assert AgentEventType.TOOL_CALL_END in types
        assert AgentEventType.ASSISTANT_TEXT_END in types

    @pytest.mark.asyncio
    async def test_turn_limit(self, env) -> None:
        """Session should stop after reaching the turn limit."""
        # Return tool calls forever
        responses = [
            _tool_call_response("shell", {"command": "echo hi"}) for _ in range(20)
        ]
        client = MockLLMClient(responses)
        config = SessionConfig(
            model_id="test-model",
            max_turns=3,
        )
        session = Session(AnthropicProfile(), env, config, client)

        events = []
        async for event in session.submit("Loop forever"):
            events.append(event)

        types = [e.type for e in events]
        assert AgentEventType.TURN_LIMIT in types

    @pytest.mark.asyncio
    async def test_session_state_transitions(self, env) -> None:
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        assert session.state == SessionState.IDLE

        async for _ in session.submit("test"):
            pass

        # After natural completion with no follow-ups, session returns to IDLE
        assert session.state == SessionState.IDLE

        await session.shutdown()
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_session_raises(self, env) -> None:
        client = MockLLMClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)
        await session.shutdown()

        with pytest.raises(RuntimeError, match="closed"):
            async for _ in session.submit("test"):
                pass

    @pytest.mark.asyncio
    async def test_conversation_history_grows(self, env) -> None:
        client = MockLLMClient([_text_response("first response")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("first input"):
            pass

        history = session.conversation_history
        # Should have exactly: user message + assistant message
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_follow_up_queue(self, env) -> None:
        captured_requests: list[Request] = []

        class CapturingClient(MockLLMClient):
            async def complete(self, request: Request) -> Response:
                captured_requests.append(request)
                return await super().complete(request)

        client = CapturingClient(
            [
                _text_response("first done"),
                _text_response("follow-up done"),
            ]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)
        session.follow_up("Do this next")

        events = []
        async for event in session.submit("Start"):
            events.append(event)

        # Should have processed both the initial and follow-up
        text_events = [e for e in events if e.type == AgentEventType.ASSISTANT_TEXT_END]
        assert len(text_events) == 2

        # Verify the LLM was called exactly twice (once for initial, once for follow-up)
        assert len(captured_requests) == 2
        # The follow-up message should appear in the second call's history
        second_request_texts = [
            m.text() for m in captured_requests[1].messages if m.role == Role.USER
        ]
        assert any("Do this next" in t for t in second_request_texts)

    @pytest.mark.asyncio
    async def test_awaiting_input_to_processing_transition(self, env) -> None:
        """Submitting from AWAITING_INPUT should transition through PROCESSING."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        # Manually set to AWAITING_INPUT to simulate model asking a question
        session._state = SessionState.AWAITING_INPUT
        assert session.state == SessionState.AWAITING_INPUT

        # Submitting input from AWAITING_INPUT should transition to PROCESSING
        # and then to IDLE after natural completion
        observed_states: list[SessionState] = []
        async for _ in session.submit("user answer"):
            observed_states.append(session.state)

        assert SessionState.PROCESSING in observed_states
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_processing_state_during_execution(self, env) -> None:
        """Session should be in PROCESSING state during execution."""
        observed_states: list[SessionState] = []

        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for event in session.submit("test"):
            observed_states.append(session.state)

        # During execution we should have seen PROCESSING at some point
        # (the SESSION_START event fires right after state = PROCESSING)
        assert SessionState.PROCESSING in observed_states
        # After natural completion, state is IDLE
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_set_reasoning_effort(self, env) -> None:
        captured_requests: list[Request] = []

        class CapturingClient(MockLLMClient):
            async def complete(self, request: Request) -> Response:
                captured_requests.append(request)
                return await super().complete(request)

        client = CapturingClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)
        session.set_reasoning_effort(ReasoningEffort.HIGH)

        async for _ in session.submit("test"):
            pass

        # Verify reasoning_effort was passed through to the LLM request
        assert len(captured_requests) == 1
        assert captured_requests[0].reasoning_effort == ReasoningEffort.HIGH


class TestSessionId:
    @pytest.mark.asyncio
    async def test_session_id_is_valid_uuid(self, env) -> None:
        """Session.id should be a valid UUID4 string assigned at creation."""
        client = MockLLMClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        # Should not raise — confirms it is a valid UUID
        parsed = uuid.UUID(session.id)
        assert str(parsed) == session.id
        assert parsed.version == 4

    @pytest.mark.asyncio
    async def test_session_id_unique_per_session(self, env) -> None:
        """Each session should get a distinct UUID."""
        client = MockLLMClient()
        config = SessionConfig(model_id="test-model")
        s1 = Session(AnthropicProfile(), env, config, client)
        s2 = Session(AnthropicProfile(), env, config, client)

        assert s1.id != s2.id

    @pytest.mark.asyncio
    async def test_session_id_in_emitted_events(self, env) -> None:
        """All events emitted by Session should carry the session's ID."""
        client = MockLLMClient([_text_response("Hello!")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events: list[AgentEvent] = []
        async for event in session.submit("Hi"):
            events.append(event)

        # SESSION_START and SESSION_END are emitted by Session._event()
        session_events = [
            e
            for e in events
            if e.type in (AgentEventType.SESSION_START, AgentEventType.SESSION_END)
        ]
        assert len(session_events) == 2
        for event in session_events:
            assert event.session_id == session.id


class TestAwaitingInput:
    @pytest.mark.asyncio
    async def test_natural_completion_transitions_to_idle(self, env) -> None:
        """After a text-only response with no follow-ups, state should be IDLE."""
        client = MockLLMClient([_text_response("What would you like me to do?")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("Help me"):
            pass

        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_idle_to_processing_on_submit(self, env) -> None:
        """Submitting from IDLE should go through PROCESSING."""
        client = MockLLMClient(
            [_text_response("What file?"), _text_response("Done.")]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        # First submit → ends in IDLE
        async for _ in session.submit("Help"):
            pass
        assert session.state == SessionState.IDLE

        # Second submit from IDLE
        observed_states: list[SessionState] = []
        async for _ in session.submit("auth.py"):
            observed_states.append(session.state)

        assert SessionState.PROCESSING in observed_states
        # Ends in IDLE again (natural completion, no follow-ups)
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_session_end_event_contains_idle_state(self, env) -> None:
        """SESSION_END event data should reflect the IDLE state after completion."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events: list[AgentEvent] = []
        async for event in session.submit("test"):
            events.append(event)

        end_events = [e for e in events if e.type == AgentEventType.SESSION_END]
        assert len(end_events) == 1
        assert end_events[0].data["state"] == SessionState.IDLE.value


class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_when_idle_sets_closed(self, env) -> None:
        """Calling abort() on an idle session transitions to CLOSED."""
        client = MockLLMClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        assert session.state == SessionState.IDLE
        session.abort()
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_abort_when_idle_after_completion_sets_closed(self, env) -> None:
        """Calling abort() from IDLE (after completion) transitions to CLOSED."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("test"):
            pass
        assert session.state == SessionState.IDLE

        session.abort()
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_abort_prevents_submit_after(self, env) -> None:
        """After abort(), subsequent submit() calls raise RuntimeError."""
        client = MockLLMClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        session.abort()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in session.submit("test"):
                pass

    @pytest.mark.asyncio
    async def test_abort_during_processing_sets_closed(self, env) -> None:
        """Calling abort() while processing should result in CLOSED after the loop finishes."""
        import asyncio

        abort_triggered = False

        class AbortableClient(MockLLMClient):
            """Client that yields to the event loop before returning."""

            async def complete(self, request: Request) -> Response:
                nonlocal abort_triggered
                # Yield control so abort() can be called
                await asyncio.sleep(0)
                if abort_triggered:
                    # Simulate the loop noticing abort and wrapping up
                    return _text_response("wrapping up")
                abort_triggered = True
                return _text_response("first response")

        client = AbortableClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events: list[AgentEvent] = []

        async def _consume() -> None:
            async for event in session.submit("Run something"):
                events.append(event)
                if event.type == AgentEventType.ASSISTANT_TEXT_END:
                    # Abort after first text response arrives
                    session.abort()

        task = asyncio.create_task(_consume())
        await asyncio.wait_for(task, timeout=3.0)

        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_abort_skips_follow_ups(self, env) -> None:
        """Abort during processing should skip queued follow-ups."""
        import asyncio

        llm_called = asyncio.Event()

        class SlowClient(MockLLMClient):
            async def complete(self, request: Request) -> Response:
                llm_called.set()
                await asyncio.sleep(0.5)
                return _text_response("first done")

        client = SlowClient()
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)
        session.follow_up("follow-up task")

        events: list[AgentEvent] = []

        async def _consume() -> None:
            async for event in session.submit("Start"):
                events.append(event)

        task = asyncio.create_task(_consume())

        # Wait until the LLM call is actually in-flight, then abort.
        # This avoids a race where a fixed sleep resolves before the
        # task enters the loop (causing abort to fire pre-emptively
        # and produce 0 events instead of 1).
        await llm_called.wait()
        session.abort()
        await asyncio.wait_for(task, timeout=3.0)

        assert session.state == SessionState.CLOSED
        # The first call completes (already in-flight when abort was
        # set — abort is only checked at loop checkpoints, not mid-await).
        # The follow-up should NOT have been processed.
        text_ends = [e for e in events if e.type == AgentEventType.ASSISTANT_TEXT_END]
        assert len(text_ends) == 1


# ---------------------------------------------------------------------------
# Gap A-C01: Session state transitions per spec (PROCESSING → IDLE)
# ---------------------------------------------------------------------------


class TestSessionStateSpecCompliance:
    """Verify session states match the spec: IDLE → PROCESSING → IDLE."""

    @pytest.mark.asyncio
    async def test_idle_after_text_only_response(self, env) -> None:
        """Natural completion (text-only, no follow-ups) returns to IDLE."""
        client = MockLLMClient([_text_response("All done.")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        assert session.state == SessionState.IDLE

        async for _ in session.submit("Do something"):
            pass

        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_idle_after_tool_call_then_text(self, env, tmp_path) -> None:
        """After tool calls followed by text completion, state returns to IDLE."""
        (tmp_path / "f.txt").write_text("data\n")
        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "f.txt"}),
                _text_response("Read the file."),
            ]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("Read f.txt"):
            pass

        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_idle_after_turn_limit(self, env) -> None:
        """Turn limit causes PROCESSING → IDLE transition."""
        responses = [
            _tool_call_response("shell", {"command": "echo x"}) for _ in range(10)
        ]
        client = MockLLMClient(responses)
        config = SessionConfig(model_id="test-model", max_turns=2)
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("Loop"):
            pass

        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_full_lifecycle_idle_processing_idle(self, env) -> None:
        """Full lifecycle: IDLE → PROCESSING (during submit) → IDLE (after)."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        assert session.state == SessionState.IDLE

        saw_processing = False
        async for _ in session.submit("test"):
            if session.state == SessionState.PROCESSING:
                saw_processing = True

        assert saw_processing, "Should have observed PROCESSING during submit"
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_multiple_submits_return_to_idle(self, env) -> None:
        """Multiple sequential submits each return to IDLE."""
        client = MockLLMClient(
            [_text_response("first"), _text_response("second"), _text_response("third")]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        for prompt in ["one", "two", "three"]:
            async for _ in session.submit(prompt):
                pass
            assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_awaiting_input_state_still_allows_submit(self, env) -> None:
        """AWAITING_INPUT (if set manually) still allows submit()."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        # Manually set AWAITING_INPUT (simulating a model question scenario)
        session._state = SessionState.AWAITING_INPUT

        async for _ in session.submit("answer"):
            pass

        # After completion, should be IDLE (not AWAITING_INPUT)
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_session_end_event_state_is_idle(self, env) -> None:
        """SESSION_END event should contain 'idle' state after natural completion."""
        client = MockLLMClient([_text_response("done")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        events: list[AgentEvent] = []
        async for event in session.submit("test"):
            events.append(event)

        end_events = [e for e in events if e.type == AgentEventType.SESSION_END]
        assert len(end_events) == 1
        assert end_events[0].data["state"] == "idle"


# ---------------------------------------------------------------------------
# GAP 1: Gemini list_dir tool integration
# ---------------------------------------------------------------------------


class TestGeminiListDir:
    """Verify that GeminiProfile's list_dir tool dispatches correctly."""

    @pytest.mark.asyncio
    async def test_gemini_list_dir_invocation(self, env, tmp_path) -> None:
        """Gemini profile should successfully dispatch list_dir tool calls."""
        from attractor.agent.profiles.gemini_profile import GeminiProfile

        (tmp_path / "example.py").write_text("print('hello')")

        client = MockLLMClient(
            [
                _tool_call_response("list_dir", {"path": str(tmp_path)}),
                _text_response("Directory listed."),
            ]
        )
        config = SessionConfig(model_id="gemini-2.0-flash")
        session = Session(GeminiProfile(), env, config, client)

        events = []
        async for event in session.submit("List the directory"):
            events.append(event)

        types = [e.type for e in events]
        assert AgentEventType.TOOL_CALL_START in types
        assert AgentEventType.TOOL_CALL_END in types
        assert AgentEventType.ASSISTANT_TEXT_END in types

        # Verify tool output contains the file name
        tool_end = next(
            e for e in events if e.type == AgentEventType.TOOL_CALL_END
        )
        assert "example.py" in tool_end.data.get("output", "")
        assert not tool_end.data.get("is_error", True)

    @pytest.mark.asyncio
    async def test_gemini_profile_has_list_dir(self) -> None:
        """GeminiProfile should include list_dir in its tool definitions."""
        from attractor.agent.profiles.gemini_profile import GeminiProfile

        profile = GeminiProfile()
        tool_names = {td.name for td in profile.tool_definitions}
        assert "list_dir" in tool_names


# ---------------------------------------------------------------------------
# GAP 2: Profile-specific shell timeout
# ---------------------------------------------------------------------------


class TestProfileTimeout:
    """Verify profile-specific default_timeout_ms is propagated."""

    def test_anthropic_profile_timeout_120s(self) -> None:
        profile = AnthropicProfile()
        assert profile.default_timeout_ms == 120_000

    def test_openai_profile_timeout_10s(self) -> None:
        from attractor.agent.profiles.openai_profile import OpenAIProfile

        profile = OpenAIProfile()
        assert profile.default_timeout_ms == 10_000

    def test_gemini_profile_timeout_10s(self) -> None:
        from attractor.agent.profiles.gemini_profile import GeminiProfile

        profile = GeminiProfile()
        assert profile.default_timeout_ms == 10_000

    @pytest.mark.asyncio
    async def test_anthropic_timeout_used_in_session(self, env) -> None:
        """Session with AnthropicProfile should use 120s default shell timeout."""
        captured_requests: list[Request] = []

        class CapturingClient(MockLLMClient):
            async def complete(self, request: Request) -> Response:
                captured_requests.append(request)
                return await super().complete(request)

        client = CapturingClient(
            [
                _tool_call_response("shell", {"command": "echo hi"}),
                _text_response("Done."),
            ]
        )
        # SessionConfig with default_command_timeout_ms=None uses profile default
        config = SessionConfig(model_id="claude-sonnet-4-20250514")
        session = Session(AnthropicProfile(), env, config, client)

        events = []
        async for event in session.submit("Run echo"):
            events.append(event)

        # The shell tool call should have used 120_000 as timeout
        tool_start = next(
            e for e in events if e.type == AgentEventType.TOOL_CALL_START
        )
        # The loop sets the timeout in the arguments before dispatch;
        # verify via TOOL_CALL_START arguments (set by loop.py:158-161)
        assert tool_start.data["arguments"].get("timeout_ms") == 120_000

    @pytest.mark.asyncio
    async def test_session_config_overrides_profile_timeout(self, env) -> None:
        """Explicit SessionConfig timeout should override profile default."""
        client = MockLLMClient(
            [
                _tool_call_response("shell", {"command": "echo hi"}),
                _text_response("Done."),
            ]
        )
        config = SessionConfig(
            model_id="claude-sonnet-4-20250514",
            default_command_timeout_ms=30_000,
        )
        session = Session(AnthropicProfile(), env, config, client)

        events = []
        async for event in session.submit("Run echo"):
            events.append(event)

        tool_start = next(
            e for e in events if e.type == AgentEventType.TOOL_CALL_START
        )
        assert tool_start.data["arguments"].get("timeout_ms") == 30_000
