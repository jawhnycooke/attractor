"""Tests for typed Turn records and Session.turns tracking."""

import time

import pytest

from attractor.agent.environment import LocalExecutionEnvironment
from attractor.agent.events import AgentEventType
from attractor.agent.profiles.anthropic_profile import AnthropicProfile
from attractor.agent.session import Session, SessionConfig
from attractor.agent.tools.registry import ToolResult
from attractor.agent.turns import (
    AssistantTurn,
    SteeringTurn,
    SystemTurn,
    ToolResultsTurn,
    Turn,
    UserTurn,
)
from attractor.llm.models import (
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    TokenUsage,
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


# ---------------------------------------------------------------------------
# Turn dataclass tests
# ---------------------------------------------------------------------------


class TestTurnDataclasses:
    def test_user_turn_creation(self) -> None:
        before = time.time()
        turn = UserTurn(content="hello")
        after = time.time()

        assert turn.content == "hello"
        assert before <= turn.timestamp <= after

    def test_user_turn_explicit_timestamp(self) -> None:
        turn = UserTurn(content="x", timestamp=1000.0)
        assert turn.timestamp == 1000.0

    def test_assistant_turn_defaults(self) -> None:
        turn = AssistantTurn(content="answer")
        assert turn.content == "answer"
        assert turn.tool_calls == []
        assert turn.reasoning is None
        assert turn.usage is None
        assert turn.response_id is None
        assert isinstance(turn.timestamp, float)

    def test_assistant_turn_with_usage(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        turn = AssistantTurn(
            content="ok",
            tool_calls=[{"name": "read_file"}],
            reasoning="thinking...",
            usage=usage,
            response_id="resp_123",
        )
        assert turn.usage is not None
        assert turn.usage.input_tokens == 10
        assert turn.response_id == "resp_123"
        assert len(turn.tool_calls) == 1

    def test_tool_results_turn_defaults(self) -> None:
        turn = ToolResultsTurn()
        assert turn.results == []
        assert isinstance(turn.timestamp, float)

    def test_tool_results_turn_with_results(self) -> None:
        results = [
            ToolResult(output="file content", is_error=False),
            ToolResult(output="error msg", is_error=True),
        ]
        turn = ToolResultsTurn(results=results)
        assert len(turn.results) == 2
        assert turn.results[0].output == "file content"
        assert turn.results[1].is_error is True

    def test_steering_turn_creation(self) -> None:
        turn = SteeringTurn(content="try a different approach")
        assert turn.content == "try a different approach"
        assert isinstance(turn.timestamp, float)

    def test_turn_type_alias(self) -> None:
        """Verify the Turn union type accepts all turn types."""
        turns: list[Turn] = [
            UserTurn(content="hi"),
            AssistantTurn(content="hello"),
            ToolResultsTurn(results=[]),
            SteeringTurn(content="steer"),
        ]
        assert len(turns) == 4
        assert isinstance(turns[0], UserTurn)
        assert isinstance(turns[1], AssistantTurn)
        assert isinstance(turns[2], ToolResultsTurn)
        assert isinstance(turns[3], SteeringTurn)


# ---------------------------------------------------------------------------
# Session.turns integration tests
# ---------------------------------------------------------------------------


class TestSessionTurns:
    @pytest.mark.asyncio
    async def test_text_response_creates_user_and_assistant_turns(
        self, env
    ) -> None:
        client = MockLLMClient([_text_response("Hello!")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("Hi"):
            pass

        turns = session.turns
        user_turns = [t for t in turns if isinstance(t, UserTurn)]
        assistant_turns = [t for t in turns if isinstance(t, AssistantTurn)]

        assert len(user_turns) == 1
        assert user_turns[0].content == "Hi"
        assert len(assistant_turns) == 1
        assert assistant_turns[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_creates_tool_results_turn(
        self, env, tmp_path
    ) -> None:
        (tmp_path / "test.txt").write_text("content\n")

        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "test.txt"}),
                _text_response("Got it"),
            ]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("Read test.txt"):
            pass

        turns = session.turns
        tool_turns = [t for t in turns if isinstance(t, ToolResultsTurn)]

        assert len(tool_turns) == 1
        assert len(tool_turns[0].results) == 1
        assert not tool_turns[0].results[0].is_error

    @pytest.mark.asyncio
    async def test_turns_match_history_order(self, env, tmp_path) -> None:
        """Turn types should follow the user → tool-results → assistant pattern."""
        (tmp_path / "f.txt").write_text("data\n")

        client = MockLLMClient(
            [
                _tool_call_response("read_file", {"path": "f.txt"}),
                _text_response("Done"),
            ]
        )
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("go"):
            pass

        turn_types = [type(t) for t in session.turns]
        # Expected order: UserTurn, ToolResultsTurn, AssistantTurn
        assert UserTurn in turn_types
        assert ToolResultsTurn in turn_types
        assert AssistantTurn in turn_types

        # UserTurn should come before ToolResultsTurn
        user_idx = turn_types.index(UserTurn)
        tool_idx = turn_types.index(ToolResultsTurn)
        assistant_idx = turn_types.index(AssistantTurn)
        assert user_idx < tool_idx < assistant_idx

    @pytest.mark.asyncio
    async def test_turns_property_returns_defensive_copy(self, env) -> None:
        client = MockLLMClient([_text_response("hi")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("hello"):
            pass

        turns1 = session.turns
        turns2 = session.turns
        assert turns1 is not turns2
        assert len(turns1) == len(turns2)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_batched(self, env, tmp_path) -> None:
        """Multiple tool calls in one round should produce one ToolResultsTurn."""
        (tmp_path / "a.txt").write_text("aaa\n")
        (tmp_path / "b.txt").write_text("bbb\n")

        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolCallContent(
                    tool_name="read_file",
                    arguments={"path": "a.txt"},
                    tool_call_id="call_a",
                ),
                ToolCallContent(
                    tool_name="read_file",
                    arguments={"path": "b.txt"},
                    tool_call_id="call_b",
                ),
            ],
        )
        multi_tool_resp = Response(
            message=msg, finish_reason=FinishReason.TOOL_USE
        )

        client = MockLLMClient([multi_tool_resp, _text_response("read both")])
        config = SessionConfig(model_id="test-model")
        session = Session(AnthropicProfile(), env, config, client)

        async for _ in session.submit("read files"):
            pass

        tool_turns = [
            t for t in session.turns if isinstance(t, ToolResultsTurn)
        ]
        assert len(tool_turns) == 1
        assert len(tool_turns[0].results) == 2


# ---------------------------------------------------------------------------
# A-C02: SystemTurn tests
# ---------------------------------------------------------------------------


class TestSystemTurn:
    def test_system_turn_creation(self) -> None:
        turn = SystemTurn(content="config changed")
        assert turn.content == "config changed"
        assert isinstance(turn.timestamp, float)

    def test_system_turn_explicit_timestamp(self) -> None:
        turn = SystemTurn(content="event", timestamp=2000.0)
        assert turn.timestamp == 2000.0

    def test_system_turn_timestamp_auto_set(self) -> None:
        before = time.time()
        turn = SystemTurn(content="auto")
        after = time.time()
        assert before <= turn.timestamp <= after

    def test_system_turn_in_turn_union(self) -> None:
        """SystemTurn should be part of the Turn union type."""
        turn: Turn = SystemTurn(content="internal")
        assert isinstance(turn, SystemTurn)

    def test_system_turn_distinct_from_steering_turn(self) -> None:
        """SystemTurn and SteeringTurn are separate types."""
        system = SystemTurn(content="system event")
        steering = SteeringTurn(content="steer")
        assert type(system) is not type(steering)

    def test_turn_type_alias_includes_system_turn(self) -> None:
        """The Turn union type should include all 5 turn types."""
        turns: list[Turn] = [
            UserTurn(content="hi"),
            AssistantTurn(content="hello"),
            ToolResultsTurn(results=[]),
            SteeringTurn(content="steer"),
            SystemTurn(content="internal"),
        ]
        assert len(turns) == 5
        assert isinstance(turns[4], SystemTurn)
