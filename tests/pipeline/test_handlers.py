"""Tests for pipeline node handlers."""

import asyncio
import builtins
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from attractor.pipeline.handlers import (
    CodergenBackend,
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    FanInHandler,
    HandlerHook,
    HandlerRegistry,
    HumanGateHandler,
    ManagerLoopHandler,
    NodeHandler,
    ParallelHandler,
    StartHandler,
    SupervisorHandler,
    ToolHandler,
    WaitHumanHandler,
    _parse_accelerator_key,
    _write_status_file,
    create_default_registry,
)
from attractor.pipeline.models import (
    NodeResult,
    OutcomeStatus,
    Pipeline,
    PipelineContext,
    PipelineEdge,
    PipelineNode,
)


def _make_node(
    name: str = "test_node",
    handler_type: str = "echo",
    **attrs: object,
) -> PipelineNode:
    return PipelineNode(name=name, handler_type=handler_type, attributes=dict(attrs))


# ---------------------------------------------------------------------------
# StartHandler
# ---------------------------------------------------------------------------


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_start_handler_succeeds(self) -> None:
        handler = StartHandler()
        node = _make_node(handler_type="start")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.notes == "Pipeline started"

    @pytest.mark.asyncio
    async def test_start_handler_accepts_graph_and_logs_root(self) -> None:
        handler = StartHandler()
        node = _make_node(handler_type="start")
        ctx = PipelineContext()
        pipeline = Pipeline(name="test")

        result = await handler.execute(node, ctx, graph=pipeline, logs_root=Path("/tmp"))
        assert result.success is True


# ---------------------------------------------------------------------------
# ExitHandler
# ---------------------------------------------------------------------------


class TestExitHandler:
    @pytest.mark.asyncio
    async def test_exit_handler_succeeds(self) -> None:
        handler = ExitHandler()
        node = _make_node(handler_type="exit")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.notes == "Pipeline exiting"


# ---------------------------------------------------------------------------
# CodergenHandler
# ---------------------------------------------------------------------------


@contextmanager
def _simulate_agent_import_error(error_cls: type[Exception] = ImportError, message: str = "no agent"):
    """Patch imports so attractor.agent.* modules raise on import."""
    real_import = builtins.__import__

    def selective_import(name, *args, **kwargs):
        if name.startswith("attractor.agent"):
            raise error_cls(message)
        return real_import(name, *args, **kwargs)

    with patch.dict("sys.modules", {"attractor.agent.events": None}):
        with patch("builtins.__import__", side_effect=selective_import):
            yield


class TestCodergenHandler:
    @pytest.mark.asyncio
    async def test_codergen_simulation_mode_on_import_error(self) -> None:
        """P-C01/P-C02: backend=None enters simulation mode (SUCCESS)."""
        handler = CodergenHandler()
        node = _make_node(handler_type="codergen", prompt="do stuff", model="gpt-4o")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert "[Simulated] Response for stage: test_node" in result.output
        assert "simulation mode" in (result.notes or "")
        assert result.context_updates["last_response"] == result.output

    @pytest.mark.asyncio
    async def test_handler_exception_is_logged(self, caplog) -> None:
        """Regression for A4: exceptions should be logged via logger.exception."""

        class FailingBackend:
            async def run(self, node, prompt, context):
                raise RuntimeError("unexpected failure")

        handler = CodergenHandler(backend=FailingBackend())
        node = _make_node(handler_type="codergen", prompt="fail", model="gpt-4o")
        ctx = PipelineContext()

        with caplog.at_level(
            logging.ERROR, logger="attractor.pipeline.handlers"
        ):
            result = await handler.execute(node, ctx)

        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "Handler failed" in caplog.text

    @pytest.mark.asyncio
    async def test_codergen_falls_back_to_node_label(self, tmp_path) -> None:
        """H3: Fall back to node.label as prompt text when node.prompt is empty."""
        handler = CodergenHandler()
        node = PipelineNode(
            name="test_node",
            handler_type="codergen",
            attributes={"model": "gpt-4o"},
            label="Fix the login bug",
        )
        ctx = PipelineContext()

        with _simulate_agent_import_error():
            await handler.execute(node, ctx, logs_root=tmp_path)

        # Prompt resolution happens BEFORE the import attempt, so we can
        # verify via the written prompt.md that the label was used.
        prompt_file = tmp_path / "test_node" / "prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "Fix the login bug"

    @pytest.mark.asyncio
    async def test_codergen_expands_goal_variable(self, tmp_path) -> None:
        """H4: $goal in prompts should be expanded from pipeline metadata."""
        handler = CodergenHandler()
        node = _make_node(
            handler_type="codergen", prompt="Work on $goal", model="gpt-4o"
        )
        ctx = PipelineContext()
        pipeline = Pipeline(
            name="test",
            goal="fix all bugs",
            metadata={"goal": "fix all bugs"},
        )

        with _simulate_agent_import_error():
            await handler.execute(
                node, ctx, graph=pipeline, logs_root=tmp_path
            )

        # Prompt expansion happens BEFORE the import attempt, so we can
        # verify via prompt.md that $goal was replaced.
        prompt_file = tmp_path / node.name / "prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "Work on fix all bugs"

    @pytest.mark.asyncio
    async def test_codergen_writes_logs(self, tmp_path) -> None:
        """H2: Write prompt.md, response.md, and status.json to logs_root."""
        handler = CodergenHandler()
        node = _make_node(
            handler_type="codergen", prompt="do stuff", model="gpt-4o"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)

        assert result.status == OutcomeStatus.SUCCESS

        # Check that prompt.md, response.md, and status.json were written
        stage_dir = tmp_path / "test_node"
        assert stage_dir.exists()
        assert (stage_dir / "prompt.md").exists()
        assert (stage_dir / "prompt.md").read_text() == "do stuff"
        assert (stage_dir / "response.md").exists()
        assert "[Simulated]" in (stage_dir / "response.md").read_text()
        assert (stage_dir / "status.json").exists()
        status = json.loads((stage_dir / "status.json").read_text())
        assert status["status"] == "success"

    @pytest.mark.asyncio
    async def test_codergen_sets_last_response_context_key(self) -> None:
        """H5: Should set 'last_response' not 'last_codergen_output'."""

        class EchoBackend:
            async def run(self, node, prompt, context):
                return "agent output"

        handler = CodergenHandler(backend=EchoBackend())
        node = _make_node(
            handler_type="codergen", prompt="do stuff", model="gpt-4o"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.status == OutcomeStatus.SUCCESS
        assert "last_response" in result.context_updates
        assert result.context_updates["last_response"] == "agent output"
        assert "last_codergen_output" not in result.context_updates


# ---------------------------------------------------------------------------
# CodergenBackend protocol
# ---------------------------------------------------------------------------


class TestCodergenBackendProtocol:
    """Tests for the P-C01 CodergenBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """CodergenBackend should be a runtime_checkable Protocol."""

        class MyBackend:
            async def run(self, node, prompt, context):
                return "output"

        assert isinstance(MyBackend(), CodergenBackend)

    def test_non_backend_is_not_instance(self) -> None:
        """Classes missing run() should not satisfy the protocol."""

        class NotABackend:
            pass

        assert not isinstance(NotABackend(), CodergenBackend)

    @pytest.mark.asyncio
    async def test_custom_backend_receives_correct_args(self) -> None:
        """Custom backend receives node, prompt, and context."""
        received: dict[str, object] = {}

        class CapturingBackend:
            async def run(self, node, prompt, context):
                received["node"] = node
                received["prompt"] = prompt
                received["context"] = context
                return "custom output"

        backend = CapturingBackend()
        handler = CodergenHandler(backend=backend)
        node = _make_node(handler_type="codergen", prompt="do stuff", model="gpt-4o")
        ctx = PipelineContext.from_dict({"key": "value"})

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.output == "custom output"
        assert result.context_updates["last_response"] == "custom output"
        assert received["node"] is node
        assert received["prompt"] == "do stuff"
        assert received["context"] is ctx

    @pytest.mark.asyncio
    async def test_failing_backend_returns_fail(self) -> None:
        """Backend exception → FAIL result."""

        class FailingBackend:
            async def run(self, node, prompt, context):
                raise RuntimeError("backend broke")

        handler = CodergenHandler(backend=FailingBackend())
        node = _make_node(handler_type="codergen", prompt="go", model="gpt-4o")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "backend broke" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_simulation_mode_output_contains_node_name(self) -> None:
        """Simulation mode includes node name in output text."""
        handler = CodergenHandler()
        node = _make_node(
            name="review_code", handler_type="codergen", prompt="review", model="gpt-4o"
        )
        ctx = PipelineContext()

        with _simulate_agent_import_error():
            result = await handler.execute(node, ctx)

        assert result.success is True
        assert "review_code" in result.output


# ---------------------------------------------------------------------------
# ToolHandler
# ---------------------------------------------------------------------------


class TestToolHandler:
    @pytest.mark.asyncio
    async def test_tool_handler_nonzero_exit_returns_failure(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="exit 1")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert result.context_updates.get("exit_code") == 1

    @pytest.mark.asyncio
    async def test_tool_handler_interpolates_context_in_command(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="echo {message}")
        ctx = PipelineContext.from_dict({"message": "hello_world"})

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert "hello_world" in result.output

    @pytest.mark.asyncio
    async def test_tool_handler_uses_tool_command_attribute(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="echo tool_cmd")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert "tool_cmd" in result.output

    @pytest.mark.asyncio
    async def test_tool_handler_stores_tool_output_in_context(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="echo hello")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert "tool.output" in result.context_updates

    @pytest.mark.asyncio
    async def test_tool_handler_no_command_returns_failure(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "No tool_command" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_tool_handler_ignores_command_attribute(self) -> None:
        """H13: Only tool_command is used, command is ignored."""
        handler = ToolHandler()
        node = _make_node(handler_type="tool", command="echo old_way")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert "No tool_command" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_tool_handler_writes_status_file_on_success(self, tmp_path) -> None:
        """#12: ToolHandler writes status.json on success."""
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="echo ok")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)
        assert result.success is True

        status_path = tmp_path / "test_node" / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["status"] == "success"
        assert status["node"] == "test_node"
        assert status["handler"] == "tool"

    @pytest.mark.asyncio
    async def test_tool_handler_writes_status_file_on_failure(self, tmp_path) -> None:
        """#12: ToolHandler writes status.json on failure."""
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="exit 1")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)
        assert result.success is False

        status_path = tmp_path / "test_node" / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["status"] == "fail"
        assert status["node"] == "test_node"
        assert status["handler"] == "tool"

    @pytest.mark.asyncio
    async def test_tool_handler_no_status_file_without_logs_root(self) -> None:
        """#12: No status.json when logs_root is None."""
        handler = ToolHandler()
        node = _make_node(handler_type="tool", tool_command="echo ok")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        # No crash, no file written (no logs_root to write to)


# ---------------------------------------------------------------------------
# WaitHumanHandler (formerly HumanGateHandler)
# ---------------------------------------------------------------------------


class TestWaitHumanHandler:
    @pytest.mark.asyncio
    async def test_wait_human_without_interviewer_fails(self) -> None:
        handler = WaitHumanHandler(interviewer=None)
        node = _make_node(handler_type="wait.human", prompt="Continue?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "No interviewer" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_wait_human_with_interviewer_approval(self) -> None:
        interviewer = AsyncMock()
        interviewer.confirm = AsyncMock(return_value=True)
        interviewer.inform = AsyncMock()

        handler = WaitHumanHandler(interviewer=interviewer)
        node = _make_node(handler_type="wait.human", prompt="Approve?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.context_updates["approved"] is True

    @pytest.mark.asyncio
    async def test_wait_human_with_interviewer_rejection(self) -> None:
        interviewer = AsyncMock()
        interviewer.confirm = AsyncMock(return_value=False)
        interviewer.inform = AsyncMock()

        handler = WaitHumanHandler(interviewer=interviewer)
        node = _make_node(handler_type="wait.human", prompt="Approve?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.context_updates["approved"] is False

    @pytest.mark.asyncio
    async def test_wait_human_with_pipeline_uses_edge_targets(self) -> None:
        """H6: Uses suggested_next_ids from outgoing edge targets and sets
        human_choice/selected_edge_id context keys."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        interviewer.ask = AsyncMock(return_value="approve")

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="approve"),
                PipelineEdge(source="gate", target="retry", label="reject"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        # H6: Uses suggested_next_ids, not preferred_label
        assert result.suggested_next_ids == ["next"]
        assert result.context_updates["human_choice"] == "approve"
        assert result.context_updates["selected_edge_id"] == "next"
        interviewer.ask.assert_called_once()

        # P-C10: Verify a Question object was passed to interviewer.ask()
        call_args = interviewer.ask.call_args
        question_arg = call_args[0][0]  # first positional arg is the Question
        # Question should contain the node's prompt text
        assert "Choose?" in str(question_arg)
        # Question options should include the edge labels
        option_labels = [o.label for o in question_arg.options]
        assert "approve" in option_labels
        assert "reject" in option_labels

    @pytest.mark.asyncio
    async def test_wait_human_uses_graph_param_over_stored_pipeline(self) -> None:
        """Handler should prefer the graph parameter over the stored pipeline."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        interviewer.ask = AsyncMock(return_value="deploy")

        stored_pipeline = Pipeline(name="stored")
        graph = Pipeline(
            name="graph",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="deploy_node", label="deploy"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=stored_pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, graph=graph)
        assert result.success is True
        assert result.suggested_next_ids == ["deploy_node"]

    @pytest.mark.asyncio
    async def test_backward_compat_alias(self) -> None:
        """HumanGateHandler alias should still work."""
        assert HumanGateHandler is WaitHumanHandler

    @pytest.mark.asyncio
    async def test_accelerator_keys_passed_to_interviewer(self) -> None:
        """H7: Accelerator keys should be parsed and passed via Question options."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        # Return the exact label for the first edge so selection works
        interviewer.ask = AsyncMock(return_value="[Y] Yes, deploy")

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(
                    source="gate", target="deploy", label="[Y] Yes, deploy"
                ),
                PipelineEdge(
                    source="gate", target="cancel", label="[N] No, cancel"
                ),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Deploy?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.suggested_next_ids == ["deploy"]

        # P-C10: Verify Question options contain accelerator keys
        call_args = interviewer.ask.call_args
        question_arg = call_args[0][0]  # Question object
        option_keys = {o.key for o in question_arg.options}
        assert option_keys == {"Y", "N"}
        assert len(question_arg.options) == 2
        # Options should include the full labels
        option_labels = {o.label for o in question_arg.options}
        assert "[Y] Yes, deploy" in option_labels
        assert "[N] No, cancel" in option_labels

    @pytest.mark.asyncio
    async def test_timeout_with_default_choice(self) -> None:
        """H8: On timeout, use human.default_choice edge target if set."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()

        # Make ask raise TimeoutError
        async def slow_ask(*args, **kwargs):
            await asyncio.sleep(10)

        interviewer.ask = slow_ask

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="approve_node", label="approve"),
                PipelineEdge(source="gate", target="reject_node", label="reject"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = PipelineNode(
            name="gate",
            handler_type="wait.human",
            attributes={
                "prompt": "Approve?",
                "human.default_choice": "approve_node",
            },
            timeout=0.01,  # Very short timeout
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.suggested_next_ids == ["approve_node"]
        assert result.context_updates["selected_edge_id"] == "approve_node"

    @pytest.mark.asyncio
    async def test_timeout_without_default_choice_returns_retry(self) -> None:
        """H8: On timeout with no default, return RETRY."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()

        async def slow_ask(*args, **kwargs):
            await asyncio.sleep(10)

        interviewer.ask = slow_ask

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="next"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = PipelineNode(
            name="gate",
            handler_type="wait.human",
            attributes={"prompt": "Approve?"},
            timeout=0.01,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.RETRY
        assert "timeout" in (result.failure_reason or "").lower()
        assert "no default" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_skipped_answer_returns_fail(self) -> None:
        """H8: SKIPPED answer should return FAIL."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        interviewer.ask = AsyncMock(return_value="SKIPPED")

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="proceed"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Approve?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "skipped" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_confirm_timeout_with_default_choice(self) -> None:
        """H8: Confirm path timeout with default choice returns SUCCESS."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()

        async def slow_confirm(*args, **kwargs):
            await asyncio.sleep(10)

        interviewer.confirm = slow_confirm

        handler = WaitHumanHandler(interviewer=interviewer)
        node = PipelineNode(
            name="gate",
            handler_type="wait.human",
            attributes={
                "prompt": "Continue?",
                "human.default_choice": "yes",
            },
            timeout=0.01,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.context_updates["approved"] is True
        assert result.context_updates["human_choice"] == "yes"

    @pytest.mark.asyncio
    async def test_confirm_timeout_without_default_returns_retry(self) -> None:
        """H8: Confirm path timeout with no default returns RETRY."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()

        async def slow_confirm(*args, **kwargs):
            await asyncio.sleep(10)

        interviewer.confirm = slow_confirm

        handler = WaitHumanHandler(interviewer=interviewer)
        node = PipelineNode(
            name="gate",
            handler_type="wait.human",
            attributes={"prompt": "Continue?"},
            timeout=0.01,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.RETRY
        assert "timeout" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_confirm_skipped_returns_fail(self) -> None:
        """H8: SKIPPED on confirm path returns FAIL."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        interviewer.confirm = AsyncMock(return_value="SKIPPED")

        handler = WaitHumanHandler(interviewer=interviewer)
        node = _make_node(handler_type="wait.human", prompt="Continue?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "skipped" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_no_timeout_when_node_timeout_is_none(self) -> None:
        """H8: No timeout wrapping when node.timeout is None."""
        interviewer = AsyncMock()
        interviewer.inform = AsyncMock()
        interviewer.ask = AsyncMock(return_value="go")

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="go"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        assert node.timeout is None  # Confirm timeout not set
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.suggested_next_ids == ["next"]


# ---------------------------------------------------------------------------
# ConditionalHandler
# ---------------------------------------------------------------------------


class TestConditionalHandler:
    @pytest.mark.asyncio
    async def test_conditional_handler_is_noop(self) -> None:
        """H10: ConditionalHandler should be a pure no-op."""
        handler = ConditionalHandler()
        node = _make_node(name="cond", handler_type="conditional")
        ctx = PipelineContext()

        pipeline = Pipeline(
            name="test",
            nodes={"cond": PipelineNode(name="cond", handler_type="conditional")},
            edges=[
                PipelineEdge(
                    source="cond", target="a", condition="x == true"
                ),
                PipelineEdge(source="cond", target="b"),
            ],
        )

        result = await handler.execute(node, ctx, graph=pipeline)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        # Should NOT set next_node — that's the engine's job
        assert result.next_node is None

    @pytest.mark.asyncio
    async def test_conditional_handler_no_constructor_args(self) -> None:
        """H10: ConditionalHandler should not require pipeline arg."""
        handler = ConditionalHandler()
        node = _make_node(handler_type="conditional")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# FanInHandler
# ---------------------------------------------------------------------------


class TestFanInHandler:
    @pytest.mark.asyncio
    async def test_fan_in_no_results(self) -> None:
        handler = FanInHandler()
        node = _make_node(handler_type="parallel.fan_in")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert "No parallel results" in result.output

    @pytest.mark.asyncio
    async def test_fan_in_with_results(self) -> None:
        handler = FanInHandler()
        node = _make_node(handler_type="parallel.fan_in")
        ctx = PipelineContext.from_dict(
            {"parallel.results": {"branch_a": "output_a", "branch_b": "output_b"}}
        )

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert "parallel.fan_in.best_id" in result.context_updates

    @pytest.mark.asyncio
    async def test_fan_in_ranks_by_status_priority(self) -> None:
        """H12: Should rank by status priority (SUCCESS > PARTIAL > FAIL)."""
        handler = FanInHandler()
        node = _make_node(handler_type="parallel.fan_in")
        ctx = PipelineContext.from_dict(
            {
                "parallel.results": {
                    "fail_branch": {"status": "fail", "output": "failed"},
                    "success_branch": {"status": "success", "output": "good"},
                    "partial_branch": {"status": "partial_success", "output": "ok"},
                }
            }
        )

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.context_updates["parallel.fan_in.best_id"] == "success_branch"


# ---------------------------------------------------------------------------
# ParallelHandler
# ---------------------------------------------------------------------------


class TestParallelHandler:
    @pytest.mark.asyncio
    async def test_parallel_handler_no_branches_returns_early(self) -> None:
        handler = ParallelHandler()
        node = _make_node(handler_type="parallel")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_parallel_handler_stores_results_in_context(self) -> None:
        """ParallelHandler should store branch outputs in context_updates."""
        registry = HandlerRegistry()
        registry.register("start", StartHandler())

        pipeline = Pipeline(
            name="test",
            nodes={
                "parallel_node": PipelineNode(
                    name="parallel_node", handler_type="parallel"
                ),
                "branch_a": PipelineNode(
                    name="branch_a", handler_type="start"
                ),
                "branch_b": PipelineNode(
                    name="branch_b", handler_type="start"
                ),
            },
            edges=[
                PipelineEdge(source="parallel_node", target="branch_a"),
                PipelineEdge(source="parallel_node", target="branch_b"),
            ],
        )

        handler = ParallelHandler(registry=registry, pipeline=pipeline)
        node = _make_node(name="parallel_node", handler_type="parallel")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, graph=pipeline)
        assert result.success is True
        assert "parallel.results" in result.context_updates
        results = result.context_updates["parallel.results"]
        assert "branch_a" in results
        assert "branch_b" in results


# ---------------------------------------------------------------------------
# ManagerLoopHandler (formerly SupervisorHandler)
# ---------------------------------------------------------------------------


class TestManagerLoopHandler:
    @pytest.mark.asyncio
    async def test_manager_loop_without_engine_fails(self) -> None:
        handler = ManagerLoopHandler(engine=None)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=3,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "No engine" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_backward_compat_alias(self) -> None:
        """SupervisorHandler alias should still work."""
        assert SupervisorHandler is ManagerLoopHandler

    @pytest.mark.asyncio
    async def test_manager_loop_sets_supervisor_feedback_on_success(self) -> None:
        """H14: Sets _supervisor_assessment and _supervisor_feedback context keys."""
        engine = AsyncMock()
        engine.run_sub_pipeline = AsyncMock()

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=2,
            done_condition="done = true",
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True

        # After iterations, supervisor context keys should be set
        assert ctx.get("_supervisor_assessment") is not None
        assert ctx.get("_supervisor_feedback") == ""
        assert "succeeded" in ctx.get("_supervisor_assessment")

    @pytest.mark.asyncio
    async def test_manager_loop_sets_feedback_on_failure(self) -> None:
        """H14: Sets failure feedback when sub-pipeline fails."""
        engine = AsyncMock()

        async def mock_run_sub(name, context):
            context.set("_sub_pipeline_status", "failed")
            context.set("_sub_pipeline_outcome", "fail")

        engine.run_sub_pipeline = mock_run_sub

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=5,
            max_consecutive_failures=3,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "3 consecutive" in (result.failure_reason or "")
        assert "failed" in ctx.get("_supervisor_assessment", "")
        assert "Sub-pipeline failed" in ctx.get("_supervisor_feedback", "")

    @pytest.mark.asyncio
    async def test_manager_loop_early_termination_on_repeated_failures(self) -> None:
        """H14: Early termination after max_consecutive_failures."""
        call_count = 0
        engine = AsyncMock()

        async def mock_run_sub(name, context):
            nonlocal call_count
            call_count += 1
            context.set("_sub_pipeline_outcome", "fail")

        engine.run_sub_pipeline = mock_run_sub

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=10,
            max_consecutive_failures=2,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert call_count == 2  # Should stop after 2 consecutive failures
        assert result.context_updates["_supervisor_iterations"] == 2

    @pytest.mark.asyncio
    async def test_manager_loop_resets_consecutive_failures_on_success(self) -> None:
        """H14: Consecutive failure count resets on successful iteration."""
        iteration = 0
        engine = AsyncMock()

        async def mock_run_sub(name, context):
            nonlocal iteration
            iteration += 1
            if iteration == 1:
                context.set("_sub_pipeline_outcome", "fail")
            elif iteration == 2:
                # Success resets consecutive failure count
                context.set("_sub_pipeline_outcome", "success")
                context.set("_sub_pipeline_status", "completed")
            elif iteration == 3:
                context.set("_sub_pipeline_outcome", "fail")
            elif iteration == 4:
                # Clear the fail outcome and set done condition
                context.set("_sub_pipeline_outcome", "success")
                context.set("done", "true")

        engine.run_sub_pipeline = mock_run_sub

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=10,
            max_consecutive_failures=2,
            done_condition="done = true",
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert iteration == 4  # All 4 iterations ran (no early termination)

    @pytest.mark.asyncio
    async def test_manager_loop_writes_status_file_on_success(self, tmp_path) -> None:
        """#12: ManagerLoopHandler writes status.json on success."""
        engine = AsyncMock()
        engine.run_sub_pipeline = AsyncMock()

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=1,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)
        assert result.success is True

        status_path = tmp_path / "test_node" / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["status"] == "success"
        assert status["node"] == "test_node"
        assert status["handler"] == "stack.manager_loop"

    @pytest.mark.asyncio
    async def test_manager_loop_writes_status_file_on_failure(self, tmp_path) -> None:
        """#12: ManagerLoopHandler writes status.json on consecutive failure."""
        engine = AsyncMock()

        async def mock_run_sub(name, context):
            context.set("_sub_pipeline_outcome", "fail")

        engine.run_sub_pipeline = mock_run_sub

        handler = ManagerLoopHandler(engine=engine)
        node = _make_node(
            handler_type="stack.manager_loop",
            sub_pipeline="inner",
            max_iterations=10,
            max_consecutive_failures=2,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)
        assert result.success is False

        status_path = tmp_path / "test_node" / "status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["status"] == "fail"
        assert "reason" in status
        assert "consecutive" in status["reason"]


# ---------------------------------------------------------------------------
# HandlerRegistry
# ---------------------------------------------------------------------------


class TestHandlerRegistry:
    def test_handler_registry_get_unknown_returns_none(self) -> None:
        registry = HandlerRegistry()
        assert registry.get("nonexistent") is None

    def test_handler_registry_default_handler_fallback(self) -> None:
        """H15: Unknown types should fall back to default_handler."""
        default = StartHandler()
        registry = HandlerRegistry(default_handler=default)
        result = registry.get("unknown_type")
        assert result is default

    def test_handler_registry_explicit_overrides_default(self) -> None:
        """Explicit registration takes priority over default."""
        default = StartHandler()
        explicit = ExitHandler()
        registry = HandlerRegistry(default_handler=default)
        registry.register("my_type", explicit)
        assert registry.get("my_type") is explicit

    def test_handler_registry_default_handler_property(self) -> None:
        """Default handler can be set/get via property."""
        registry = HandlerRegistry()
        assert registry.default_handler is None
        handler = StartHandler()
        registry.default_handler = handler
        assert registry.default_handler is handler

    def test_handler_registry_hooks_init_empty(self) -> None:
        """#16: Hooks list defaults to empty."""
        registry = HandlerRegistry()
        assert registry.hooks == []

    def test_handler_registry_hooks_init_with_list(self) -> None:
        """#16: Hooks can be passed at construction time."""
        hook = AsyncMock(spec=HandlerHook)
        registry = HandlerRegistry(hooks=[hook])
        assert len(registry.hooks) == 1

    def test_handler_registry_add_hook(self) -> None:
        """#16: Hooks can be added after construction."""
        registry = HandlerRegistry()
        hook = AsyncMock(spec=HandlerHook)
        registry.add_hook(hook)
        assert len(registry.hooks) == 1

    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self) -> None:
        """#16: dispatch() should call the correct handler."""
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        node = _make_node(handler_type="start")
        ctx = PipelineContext()

        result = await registry.dispatch("start", node, ctx)
        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_dispatch_returns_fail_for_unknown_type(self) -> None:
        """#16: dispatch() returns FAIL for unregistered handler types."""
        registry = HandlerRegistry()
        node = _make_node(handler_type="unknown")
        ctx = PipelineContext()

        result = await registry.dispatch("unknown", node, ctx)
        assert result.success is False
        assert "No handler" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_dispatch_calls_hooks_in_order(self) -> None:
        """#16: dispatch() calls before_execute and after_execute hooks."""
        call_order: list[str] = []

        class TrackingHook:
            def __init__(self, name: str) -> None:
                self._name = name

            async def before_execute(self, node, context):
                call_order.append(f"{self._name}_before")

            async def after_execute(self, node, context, result):
                call_order.append(f"{self._name}_after")

        hook1 = TrackingHook("hook1")
        hook2 = TrackingHook("hook2")

        registry = HandlerRegistry(hooks=[hook1, hook2])
        registry.register("start", StartHandler())
        node = _make_node(handler_type="start")
        ctx = PipelineContext()

        result = await registry.dispatch("start", node, ctx)
        assert result.success is True
        assert call_order == [
            "hook1_before",
            "hook2_before",
            "hook1_after",
            "hook2_after",
        ]

    @pytest.mark.asyncio
    async def test_dispatch_hook_receives_result(self) -> None:
        """#16: after_execute hook receives the handler result."""
        captured_results: list[NodeResult] = []

        class CapturingHook:
            async def before_execute(self, node, context):
                pass

            async def after_execute(self, node, context, result):
                captured_results.append(result)

        registry = HandlerRegistry(hooks=[CapturingHook()])
        registry.register("start", StartHandler())
        node = _make_node(handler_type="start")
        ctx = PipelineContext()

        result = await registry.dispatch("start", node, ctx)
        assert len(captured_results) == 1
        assert captured_results[0] is result
        assert captured_results[0].status == OutcomeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_dispatch_no_hooks_still_works(self) -> None:
        """#16: dispatch() works correctly without any hooks."""
        registry = HandlerRegistry()
        registry.register("exit", ExitHandler())
        node = _make_node(handler_type="exit")
        ctx = PipelineContext()

        result = await registry.dispatch("exit", node, ctx)
        assert result.success is True
        assert result.notes == "Pipeline exiting"


# ---------------------------------------------------------------------------
# Accelerator key parsing
# ---------------------------------------------------------------------------


class TestParseAcceleratorKey:
    def test_bracket_pattern(self) -> None:
        assert _parse_accelerator_key("[Y] Yes, deploy") == "Y"

    def test_paren_pattern(self) -> None:
        assert _parse_accelerator_key("Y) Yes, deploy") == "Y"

    def test_dash_pattern(self) -> None:
        assert _parse_accelerator_key("Y - Yes, deploy") == "Y"

    def test_first_char_fallback(self) -> None:
        assert _parse_accelerator_key("Yes, deploy") == "Y"

    def test_empty_string(self) -> None:
        assert _parse_accelerator_key("") == ""


# ---------------------------------------------------------------------------
# _write_status_file
# ---------------------------------------------------------------------------


class TestWriteStatusFile:
    def test_writes_success_status(self, tmp_path) -> None:
        """#12: _write_status_file writes correct success format."""
        node = _make_node(name="my_node", handler_type="tool")
        _write_status_file(tmp_path, node, "success")

        status_path = tmp_path / "my_node" / "status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text())
        assert data["status"] == "success"
        assert data["node"] == "my_node"
        assert data["handler"] == "tool"
        assert "reason" not in data

    def test_writes_fail_status_with_reason(self, tmp_path) -> None:
        """#12: _write_status_file includes reason on failure."""
        node = _make_node(name="my_node", handler_type="codergen")
        _write_status_file(tmp_path, node, "fail", "something broke")

        data = json.loads(
            (tmp_path / "my_node" / "status.json").read_text()
        )
        assert data["status"] == "fail"
        assert data["reason"] == "something broke"
        assert data["handler"] == "codergen"

    def test_creates_stage_directory(self, tmp_path) -> None:
        """#12: _write_status_file creates the stage directory if needed."""
        node = _make_node(name="new_stage", handler_type="tool")
        _write_status_file(tmp_path, node, "success")

        assert (tmp_path / "new_stage").is_dir()
        assert (tmp_path / "new_stage" / "status.json").exists()


# ---------------------------------------------------------------------------
# HandlerHook protocol
# ---------------------------------------------------------------------------


class TestHandlerHookProtocol:
    def test_handler_hook_is_runtime_checkable(self) -> None:
        """#16: HandlerHook should be a runtime_checkable Protocol."""

        class MyHook:
            async def before_execute(self, node, context):
                pass

            async def after_execute(self, node, context, result):
                pass

        assert isinstance(MyHook(), HandlerHook)

    def test_non_hook_is_not_handler_hook(self) -> None:
        """#16: Classes missing methods should not pass isinstance check."""

        class NotAHook:
            pass

        assert not isinstance(NotAHook(), HandlerHook)

    def test_node_handler_protocol_is_runtime_checkable(self) -> None:
        """NodeHandler should be a runtime_checkable Protocol."""
        assert isinstance(StartHandler(), NodeHandler)
        assert isinstance(ExitHandler(), NodeHandler)


# ---------------------------------------------------------------------------
# create_default_registry
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    def test_all_spec_types_registered(self) -> None:
        registry = create_default_registry()
        expected_types = [
            "start",
            "exit",
            "codergen",
            "wait.human",
            "conditional",
            "parallel",
            "parallel.fan_in",
            "tool",
            "stack.manager_loop",
        ]
        for handler_type in expected_types:
            assert registry.has(handler_type), f"Missing handler: {handler_type}"

    def test_default_handler_is_set(self) -> None:
        """H15: Default handler should be CodergenHandler."""
        registry = create_default_registry()
        assert registry.default_handler is not None
        assert isinstance(registry.default_handler, CodergenHandler)

    def test_unknown_type_returns_default(self) -> None:
        """H15: Unknown types should return the default handler."""
        registry = create_default_registry()
        handler = registry.get("unknown_type")
        assert handler is not None
        assert isinstance(handler, CodergenHandler)


# ---------------------------------------------------------------------------
# P-C02: Simulation mode for CodergenHandler (spec §4.5)
# ---------------------------------------------------------------------------


class TestCodergenSimulationMode:
    """Tests for P-C02: When backend is None, handler returns simulated response."""

    @pytest.mark.asyncio
    async def test_simulation_mode_when_backend_is_none(self) -> None:
        """P-C02: backend=None should directly enter simulation mode."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            handler_type="codergen", prompt="do stuff", model="gpt-4o"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.status == OutcomeStatus.SUCCESS
        assert result.output == "[Simulated] Response for stage: test_node"
        assert result.context_updates["last_response"] == result.output
        assert "simulation mode" in (result.notes or "")

    @pytest.mark.asyncio
    async def test_simulation_text_format_matches_spec(self) -> None:
        """P-C02: Simulation text must be '[Simulated] Response for stage: {node.id}'."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            name="review_step", handler_type="codergen", prompt="review"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.output == "[Simulated] Response for stage: review_step"
        assert result.output.startswith("[Simulated]")

    @pytest.mark.asyncio
    async def test_simulation_mode_writes_logs(self, tmp_path) -> None:
        """P-C02: Simulation mode writes prompt.md, response.md, status.json."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            handler_type="codergen", prompt="build feature", model="gpt-4o"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx, logs_root=tmp_path)

        assert result.success is True
        stage_dir = tmp_path / "test_node"
        assert stage_dir.exists()
        assert (stage_dir / "prompt.md").read_text() == "build feature"
        assert (stage_dir / "response.md").read_text() == (
            "[Simulated] Response for stage: test_node"
        )
        status = json.loads((stage_dir / "status.json").read_text())
        assert status["status"] == "success"
        assert status["node"] == "test_node"
        assert status["notes"] == "simulation mode"

    @pytest.mark.asyncio
    async def test_simulation_mode_no_logs_without_logs_root(self) -> None:
        """P-C02: No log files when logs_root is not provided."""
        handler = CodergenHandler(backend=None)
        node = _make_node(handler_type="codergen", prompt="test")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.output.startswith("[Simulated]")

    @pytest.mark.asyncio
    async def test_simulation_mode_expands_goal(self) -> None:
        """P-C02: $goal expansion should happen before simulation."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            handler_type="codergen", prompt="Work on $goal"
        )
        ctx = PipelineContext()
        pipeline = Pipeline(
            name="test",
            goal="fix all bugs",
            metadata={"goal": "fix all bugs"},
        )

        result = await handler.execute(node, ctx, graph=pipeline)

        assert result.success is True
        # Simulation response doesn't contain the expanded prompt,
        # but the prompt expansion still happens for logging
        assert result.output == "[Simulated] Response for stage: test_node"

    @pytest.mark.asyncio
    async def test_simulation_mode_expands_prompt_to_log(self, tmp_path) -> None:
        """P-C02: Prompt expansion writes to log even in simulation mode."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            handler_type="codergen", prompt="Work on $goal"
        )
        ctx = PipelineContext()
        pipeline = Pipeline(
            name="test",
            goal="fix all bugs",
            metadata={"goal": "fix all bugs"},
        )

        await handler.execute(node, ctx, graph=pipeline, logs_root=tmp_path)

        prompt_file = tmp_path / "test_node" / "prompt.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "Work on fix all bugs"

    @pytest.mark.asyncio
    async def test_explicit_backend_skips_simulation(self) -> None:
        """P-C02: Explicit backend should NOT enter simulation mode."""

        class EchoBackend:
            async def run(self, node, prompt, context):
                return f"real: {prompt}"

        handler = CodergenHandler(backend=EchoBackend())
        node = _make_node(
            handler_type="codergen", prompt="do work", model="gpt-4o"
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.output == "real: do work"
        assert "[Simulated]" not in result.output

    @pytest.mark.asyncio
    async def test_simulation_mode_context_interpolation(self, tmp_path) -> None:
        """P-C02: Context variables should be interpolated before simulation."""
        handler = CodergenHandler(backend=None)
        node = _make_node(
            handler_type="codergen", prompt="Fix {issue_id}"
        )
        ctx = PipelineContext.from_dict({"issue_id": "BUG-123"})

        await handler.execute(node, ctx, logs_root=tmp_path)

        prompt_file = tmp_path / "test_node" / "prompt.md"
        assert prompt_file.read_text() == "Fix BUG-123"


# ---------------------------------------------------------------------------
# P-C10: WaitHumanHandler Question/Answer interface (spec §4.6, §6.2, §6.3)
# ---------------------------------------------------------------------------


class TestWaitHumanQuestionAnswer:
    """Tests for P-C10: WaitHumanHandler uses structured Question/Answer types."""

    @pytest.mark.asyncio
    async def test_ask_receives_question_object(self) -> None:
        """P-C10: interviewer.ask() receives a Question, not raw strings."""
        from attractor.pipeline.interviewer import (
            Answer,
            AnswerValue,
            Option,
            Question,
            QuestionType,
            QueueInterviewer,
        )

        interviewer = QueueInterviewer()
        # Queue an Answer that selects the first option
        interviewer.responses.put_nowait(
            Answer(value="a", selected_option=Option(key="a", label="approve"))
        )

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="approve"),
                PipelineEdge(source="gate", target="retry", label="reject"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.suggested_next_ids == ["next"]
        assert result.context_updates["human_choice"] == "approve"
        assert result.context_updates["selected_edge_id"] == "next"

    @pytest.mark.asyncio
    async def test_question_has_multiple_choice_type(self) -> None:
        """P-C10: Question should have type=MULTIPLE_CHOICE."""
        from attractor.pipeline.interviewer import (
            Answer,
            Option,
            Question,
            QuestionType,
        )

        captured_questions: list[Question] = []

        class CapturingInterviewer:
            async def ask(self, question: Question) -> Answer:
                captured_questions.append(question)
                return Answer(
                    value="a",
                    selected_option=Option(key="a", label="approve"),
                )

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="approve"),
                PipelineEdge(source="gate", target="retry", label="reject"),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=CapturingInterviewer(), pipeline=pipeline
        )
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert len(captured_questions) == 1
        q = captured_questions[0]
        assert q.type == QuestionType.MULTIPLE_CHOICE
        assert q.text == "Choose?"
        assert q.stage == "gate"
        assert len(q.options) == 2
        assert q.options[0].label == "approve"
        assert q.options[1].label == "reject"

    @pytest.mark.asyncio
    async def test_question_options_have_accelerator_keys(self) -> None:
        """P-C10: Option.key should be the parsed accelerator key."""
        from attractor.pipeline.interviewer import (
            Answer,
            Option,
            Question,
            QuestionType,
        )

        captured_questions: list[Question] = []

        class CapturingInterviewer:
            async def ask(self, question: Question) -> Answer:
                captured_questions.append(question)
                return Answer(
                    value="Y",
                    selected_option=Option(key="Y", label="[Y] Yes, deploy"),
                )

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(
                    source="gate", target="deploy", label="[Y] Yes, deploy"
                ),
                PipelineEdge(
                    source="gate", target="cancel", label="[N] No, cancel"
                ),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=CapturingInterviewer(), pipeline=pipeline
        )
        node = _make_node(name="gate", handler_type="wait.human", prompt="Deploy?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        q = captured_questions[0]
        keys = {opt.key for opt in q.options}
        assert keys == {"Y", "N"}
        labels = {opt.label for opt in q.options}
        assert "[Y] Yes, deploy" in labels
        assert "[N] No, cancel" in labels

    @pytest.mark.asyncio
    async def test_answer_selected_option_matches_edge(self) -> None:
        """P-C10: When Answer has selected_option, edge is matched by label."""
        from attractor.pipeline.interviewer import Answer, Option

        class OptionInterviewer:
            async def ask(self, question):
                return Answer(
                    value="r",
                    selected_option=Option(key="r", label="reject"),
                )

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="approve"),
                PipelineEdge(source="gate", target="retry", label="reject"),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=OptionInterviewer(), pipeline=pipeline
        )
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.suggested_next_ids == ["retry"]
        assert result.context_updates["human_choice"] == "reject"
        assert result.context_updates["selected_edge_id"] == "retry"

    @pytest.mark.asyncio
    async def test_answer_skipped_returns_fail(self) -> None:
        """P-C10: Answer with AnswerValue.SKIPPED returns FAIL."""
        from attractor.pipeline.interviewer import Answer, AnswerValue

        class SkipInterviewer:
            async def ask(self, question):
                return Answer(value=AnswerValue.SKIPPED)

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="next", label="go"),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=SkipInterviewer(), pipeline=pipeline
        )
        node = _make_node(name="gate", handler_type="wait.human", prompt="Go?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "skipped" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_answer_no_match_falls_back_to_first_edge(self) -> None:
        """P-C10: Unmatched answer falls back to first edge."""
        from attractor.pipeline.interviewer import Answer

        class NoMatchInterviewer:
            async def ask(self, question):
                return Answer(value="unknown_option")

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="default_next", label="first"),
                PipelineEdge(source="gate", target="other", label="second"),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=NoMatchInterviewer(), pipeline=pipeline
        )
        node = _make_node(name="gate", handler_type="wait.human", prompt="Choose?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.suggested_next_ids == ["default_next"]

    @pytest.mark.asyncio
    async def test_question_stage_set_to_node_name(self) -> None:
        """P-C10: Question.stage should be set to the node name."""
        from attractor.pipeline.interviewer import Answer, Option, Question

        captured_questions: list[Question] = []

        class StageCapture:
            async def ask(self, question: Question) -> Answer:
                captured_questions.append(question)
                return Answer(
                    value="ok",
                    selected_option=Option(key="o", label="ok"),
                )

            async def inform(self, message: str, stage: str = "") -> None:
                pass

        pipeline = Pipeline(
            name="test",
            nodes={
                "my_gate_node": PipelineNode(
                    name="my_gate_node", handler_type="wait.human"
                )
            },
            edges=[
                PipelineEdge(
                    source="my_gate_node", target="next", label="ok"
                ),
            ],
        )

        handler = WaitHumanHandler(
            interviewer=StageCapture(), pipeline=pipeline
        )
        node = _make_node(
            name="my_gate_node", handler_type="wait.human", prompt="Ready?"
        )
        ctx = PipelineContext()

        await handler.execute(node, ctx)

        assert len(captured_questions) == 1
        assert captured_questions[0].stage == "my_gate_node"

    @pytest.mark.asyncio
    async def test_queue_interviewer_integration(self) -> None:
        """P-C10: Full integration with QueueInterviewer from interviewer.py."""
        from attractor.pipeline.interviewer import (
            Answer,
            AnswerValue,
            Option,
            QueueInterviewer,
        )

        interviewer = QueueInterviewer()
        # Pre-queue an answer selecting the second option
        interviewer.responses.put_nowait(
            Answer(
                value="r",
                selected_option=Option(key="r", label="reject"),
            )
        )

        pipeline = Pipeline(
            name="test",
            nodes={"gate": PipelineNode(name="gate", handler_type="wait.human")},
            edges=[
                PipelineEdge(source="gate", target="approve_node", label="approve"),
                PipelineEdge(source="gate", target="reject_node", label="reject"),
            ],
        )

        handler = WaitHumanHandler(interviewer=interviewer, pipeline=pipeline)
        node = _make_node(name="gate", handler_type="wait.human", prompt="Review?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)

        assert result.success is True
        assert result.suggested_next_ids == ["reject_node"]
        assert result.context_updates["human_choice"] == "reject"
        assert result.context_updates["selected_edge_id"] == "reject_node"
        # Verify inform was called
        assert len(interviewer.messages) == 1
        assert "gate" in interviewer.messages[0]
