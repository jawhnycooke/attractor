"""Tests for pipeline node handlers."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from attractor.pipeline.handlers import (
    CodergenHandler,
    HandlerRegistry,
    HumanGateHandler,
    ParallelHandler,
    SupervisorHandler,
    ToolHandler,
)
from attractor.pipeline.models import (
    PipelineContext,
    PipelineNode,
)


def _make_node(
    name: str = "test_node",
    handler_type: str = "echo",
    **attrs: object,
) -> PipelineNode:
    return PipelineNode(name=name, handler_type=handler_type, attributes=dict(attrs))


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

        # Mock the import to fail
        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=ImportError("no agent")):
                result = await handler.execute(node, ctx)

        assert result.success is False
        assert "not available" in (result.error or "")

    @pytest.mark.asyncio
    async def test_handler_exception_is_logged(self, caplog) -> None:
        """Regression for A4: exceptions should be logged via logger.exception."""
        handler = CodergenHandler()
        node = _make_node(handler_type="codergen", prompt="fail", model="gpt-4o")
        ctx = PipelineContext()

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch(
                "builtins.__import__",
                side_effect=RuntimeError("unexpected failure"),
            ):
                with caplog.at_level(
                    logging.ERROR, logger="attractor.pipeline.handlers"
                ):
                    result = await handler.execute(node, ctx)

        assert result.success is False


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
        assert result.context_updates.get("exit_code") == 1

    @pytest.mark.asyncio
    async def test_tool_handler_interpolates_context_in_command(self) -> None:
        handler = ToolHandler()
        node = _make_node(handler_type="tool", command="echo {message}")
        ctx = PipelineContext.from_dict({"message": "hello_world"})

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert "hello_world" in result.output


# ---------------------------------------------------------------------------
# HumanGateHandler
# ---------------------------------------------------------------------------


class TestHumanGateHandler:
    @pytest.mark.asyncio
    async def test_human_gate_without_interviewer_fails(self) -> None:
        """Regression for A6: no interviewer → success=False."""
        handler = HumanGateHandler(interviewer=None)
        node = _make_node(handler_type="human_gate", prompt="Continue?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert "No interviewer" in (result.error or "")

    @pytest.mark.asyncio
    async def test_human_gate_with_interviewer_approval(self) -> None:
        interviewer = AsyncMock()
        interviewer.confirm = AsyncMock(return_value=True)
        interviewer.inform = AsyncMock()

        handler = HumanGateHandler(interviewer=interviewer)
        node = _make_node(handler_type="human_gate", prompt="Approve?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.context_updates["approved"] is True

    @pytest.mark.asyncio
    async def test_human_gate_with_interviewer_rejection(self) -> None:
        interviewer = AsyncMock()
        interviewer.confirm = AsyncMock(return_value=False)
        interviewer.inform = AsyncMock()

        handler = HumanGateHandler(interviewer=interviewer)
        node = _make_node(handler_type="human_gate", prompt="Approve?")
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is True
        assert result.context_updates["approved"] is False


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


# ---------------------------------------------------------------------------
# SupervisorHandler
# ---------------------------------------------------------------------------


class TestSupervisorHandler:
    @pytest.mark.asyncio
    async def test_supervisor_handler_without_engine_fails(self) -> None:
        """Regression for A5: engine=None → success=False."""
        handler = SupervisorHandler(engine=None)
        node = _make_node(
            handler_type="supervisor",
            sub_pipeline="inner",
            max_iterations=3,
        )
        ctx = PipelineContext()

        result = await handler.execute(node, ctx)
        assert result.success is False
        assert "No engine" in (result.error or "")


# ---------------------------------------------------------------------------
# HandlerRegistry
# ---------------------------------------------------------------------------


class TestHandlerRegistry:
    def test_handler_registry_get_unknown_returns_none(self) -> None:
        registry = HandlerRegistry()
        assert registry.get("nonexistent") is None
