"""Tests for pipeline node handlers."""

import builtins
import logging
from unittest.mock import AsyncMock, patch

import pytest

from attractor.pipeline.handlers import (
    CodergenHandler,
    ExitHandler,
    FanInHandler,
    HandlerRegistry,
    HumanGateHandler,
    ManagerLoopHandler,
    ParallelHandler,
    StartHandler,
    SupervisorHandler,
    ToolHandler,
    WaitHumanHandler,
    create_default_registry,
)
from attractor.pipeline.models import (
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


class TestCodergenHandler:
    @pytest.mark.asyncio
    async def test_codergen_stub_returns_failure_not_success(self) -> None:
        """Regression for A5: ImportError fallback should return success=False."""
        handler = CodergenHandler()
        node = _make_node(handler_type="codergen", prompt="do stuff", model="gpt-4o")
        ctx = PipelineContext()

        # Mock the import to fail only for attractor.agent.* modules
        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name.startswith("attractor.agent"):
                raise ImportError("no agent")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = await handler.execute(node, ctx)

        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert "not available" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_handler_exception_is_logged(self, caplog) -> None:
        """Regression for A4: exceptions should be logged via logger.exception."""
        handler = CodergenHandler()
        node = _make_node(handler_type="codergen", prompt="fail", model="gpt-4o")
        ctx = PipelineContext()

        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name.startswith("attractor.agent"):
                raise RuntimeError("unexpected failure")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                with caplog.at_level(
                    logging.ERROR, logger="attractor.pipeline.handlers"
                ):
                    result = await handler.execute(node, ctx)

        assert result.success is False
        assert result.status == OutcomeStatus.FAIL


# ---------------------------------------------------------------------------
# ToolHandler
# ---------------------------------------------------------------------------


class TestToolHandler:
    @pytest.mark.asyncio
    async def test_tool_handler_nonzero_exit_returns_failure(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", command="exit 1")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert result.status == OutcomeStatus.FAIL
        assert result.context_updates.get("exit_code") == 1

    @pytest.mark.asyncio
    async def test_tool_handler_interpolates_context_in_command(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", command="echo {message}")
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
        node = _make_node(handler_type="tool", command="echo hello")
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
        assert "No command" in (result.failure_reason or "")


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
    async def test_wait_human_with_pipeline_uses_edge_labels(self) -> None:
        """When pipeline is available, derives choices from outgoing edges."""
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
        assert result.preferred_label == "approve"
        interviewer.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_backward_compat_alias(self) -> None:
        """HumanGateHandler alias should still work."""
        assert HumanGateHandler is WaitHumanHandler


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
        handler = ParallelHandler()
        node = _make_node(handler_type="parallel")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True


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


# ---------------------------------------------------------------------------
# HandlerRegistry
# ---------------------------------------------------------------------------


class TestHandlerRegistry:
    def test_handler_registry_get_unknown_returns_none(self) -> None:
        registry = HandlerRegistry()
        assert registry.get("nonexistent") is None


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
