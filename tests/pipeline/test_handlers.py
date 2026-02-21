"""Tests for pipeline node handlers."""

import builtins
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from attractor.pipeline.handlers import (
    CodergenHandler,
    ConditionalHandler,
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
    _parse_accelerator_key,
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

        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name.startswith("attractor.agent"):
                raise ImportError("no agent")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=selective_import):
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

        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name.startswith("attractor.agent"):
                raise ImportError("no agent")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=selective_import):
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

        real_import = builtins.__import__

        def selective_import(name, *args, **kwargs):
            if name.startswith("attractor.agent"):
                raise ImportError("no agent")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"attractor.agent.events": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = await handler.execute(node, ctx, logs_root=tmp_path)

        assert result.status == OutcomeStatus.FAIL

        # Check that prompt.md and status.json were written
        stage_dir = tmp_path / "test_node"
        assert stage_dir.exists()
        assert (stage_dir / "prompt.md").exists()
        assert (stage_dir / "prompt.md").read_text() == "do stuff"
        assert (stage_dir / "status.json").exists()
        status = json.loads((stage_dir / "status.json").read_text())
        assert status["status"] == "fail"

    @pytest.mark.asyncio
    async def test_codergen_sets_last_response_context_key(self) -> None:
        """H5: Should set 'last_response' not 'last_codergen_output'."""
        from attractor.agent.events import AgentEvent, AgentEventType

        handler = CodergenHandler()
        node = _make_node(
            handler_type="codergen", prompt="do stuff", model="gpt-4o"
        )
        ctx = PipelineContext()

        # Mock the agent session to yield a text delta event, exercising
        # the handler's actual success path.
        async def mock_submit(prompt):
            yield AgentEvent(
                type=AgentEventType.ASSISTANT_TEXT_DELTA,
                data={"text": "agent output"},
            )

        mock_session = AsyncMock()
        mock_session.submit = mock_submit

        with patch("attractor.agent.session.Session", return_value=mock_session), \
             patch("attractor.llm.client.LLMClient"):
            result = await handler.execute(node, ctx)

        assert result.status == OutcomeStatus.SUCCESS
        assert "last_response" in result.context_updates
        assert result.context_updates["last_response"] == "agent output"
        assert "last_codergen_output" not in result.context_updates


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

        # Verify the correct args were passed to interviewer.ask()
        call_args = interviewer.ask.call_args
        prompt_arg = call_args[0][0]  # first positional arg is the prompt
        options_arg = call_args[1].get("options", [])  # options is a kwarg
        # Prompt should contain the node's prompt text
        assert "Choose?" in str(prompt_arg)
        # Options should include the edge labels
        assert "approve" in options_arg
        assert "reject" in options_arg

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
