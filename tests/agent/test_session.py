"""Tests for Session lifecycle with a mocked LLM client."""

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.events import AgentEventType
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
        # Should have: user message + assistant message
        assert len(history) >= 2

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
        assert len(text_events) >= 2

        # Verify the LLM was called twice (once for initial, once for follow-up)
        assert len(captured_requests) >= 2
        # The follow-up message should appear in the second call's history
        second_request_texts = [
            m.text() for m in captured_requests[1].messages if m.role == Role.USER
        ]
        assert any("Do this next" in t for t in second_request_texts)

    @pytest.mark.asyncio
    async def test_awaiting_input_state_transition(self, env) -> None:
        """AWAITING_INPUT should be a valid state that transitions back to PROCESSING."""
        client = MockLLMClient([_text_response("ok")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        # Verify AWAITING_INPUT is a valid state
        assert SessionState.AWAITING_INPUT.value == "awaiting_input"

        # Manually set to AWAITING_INPUT to simulate model asking a question
        session._state = SessionState.AWAITING_INPUT
        assert session.state == SessionState.AWAITING_INPUT

        # Submitting input from AWAITING_INPUT should transition to PROCESSING
        # and eventually back to IDLE after completion
        async for _ in session.submit("user answer"):
            pass

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
        # After completion, state should be IDLE
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_set_reasoning_effort(self, env) -> None:
        from attractor.llm.models import ReasoningEffort

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
        assert len(captured_requests) >= 1
        assert captured_requests[0].reasoning_effort == ReasoningEffort.HIGH
