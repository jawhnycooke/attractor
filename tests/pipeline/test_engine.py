"""Tests for the pipeline execution engine."""

import pytest

from attractor.pipeline.engine import EngineError, PipelineEngine
from attractor.pipeline.handlers import HandlerRegistry
from attractor.pipeline.models import (
    NodeResult,
    OutcomeStatus,
    Pipeline,
    PipelineContext,
    PipelineEdge,
    PipelineNode,
)


class EchoHandler:
    """Test handler that records execution and echoes the node name."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        self.executed.append(node.name)
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            output=f"echo:{node.name}",
            context_updates={f"{node.name}_done": True},
        )


class FailHandler:
    """Test handler that always fails."""

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        return NodeResult(status=OutcomeStatus.FAIL, failure_reason="intentional failure")


class RoutingHandler:
    """Test handler that routes to a specific next node."""

    def __init__(self, target: str) -> None:
        self._target = target

    async def execute(self, node: PipelineNode, context: PipelineContext) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, next_node=self._target)


def _simple_pipeline() -> Pipeline:
    """Linear pipeline: start -> middle -> end."""
    return Pipeline(
        name="test",
        nodes={
            "start": PipelineNode(name="start", handler_type="echo", is_start=True),
            "middle": PipelineNode(name="middle", handler_type="echo"),
            "end": PipelineNode(name="end", handler_type="echo", is_terminal=True),
        },
        edges=[
            PipelineEdge(source="start", target="middle"),
            PipelineEdge(source="middle", target="end"),
        ],
        start_node="start",
    )


def _branching_pipeline() -> Pipeline:
    """Pipeline with conditional branching."""
    return Pipeline(
        name="branch",
        nodes={
            "start": PipelineNode(name="start", handler_type="echo", is_start=True),
            "left": PipelineNode(name="left", handler_type="echo", is_terminal=True),
            "right": PipelineNode(name="right", handler_type="echo", is_terminal=True),
        },
        edges=[
            PipelineEdge(
                source="start", target="left", condition="go_left == true", weight=1
            ),
            PipelineEdge(
                source="start", target="right", condition="go_left == false", weight=0
            ),
        ],
        start_node="start",
    )


class TestPipelineEngine:
    @pytest.fixture
    def echo_handler(self) -> EchoHandler:
        return EchoHandler()

    @pytest.fixture
    def registry(self, echo_handler: EchoHandler) -> HandlerRegistry:
        reg = HandlerRegistry()
        reg.register("echo", echo_handler)
        return reg

    async def test_linear_execution(
        self, registry: HandlerRegistry, echo_handler: EchoHandler
    ) -> None:
        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(_simple_pipeline())
        assert echo_handler.executed == ["start", "middle", "end"]
        assert ctx.get("start_done") is True
        assert ctx.get("end_done") is True

    async def test_context_propagation(self, registry: HandlerRegistry) -> None:
        engine = PipelineEngine(registry=registry)
        initial = PipelineContext.from_dict({"initial_key": "hello"})
        ctx = await engine.run(_simple_pipeline(), context=initial)
        assert ctx.get("initial_key") == "hello"
        assert ctx.get("start_done") is True

    async def test_conditional_branching_left(self, registry: HandlerRegistry) -> None:
        engine = PipelineEngine(registry=registry)
        ctx = PipelineContext.from_dict({"go_left": True})
        result = await engine.run(_branching_pipeline(), context=ctx)
        assert result.get("left_done") is True
        assert result.get("right_done") is None

    async def test_conditional_branching_right(self, registry: HandlerRegistry) -> None:
        engine = PipelineEngine(registry=registry)
        ctx = PipelineContext.from_dict({"go_left": False})
        result = await engine.run(_branching_pipeline(), context=ctx)
        assert result.get("right_done") is True
        assert result.get("left_done") is None

    async def test_explicit_routing_override(self) -> None:
        registry = HandlerRegistry()
        registry.register("router", RoutingHandler("end"))
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="route_test",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="router", is_start=True
                ),
                "middle": PipelineNode(name="middle", handler_type="echo"),
                "end": PipelineNode(name="end", handler_type="echo", is_terminal=True),
            },
            edges=[
                PipelineEdge(source="start", target="middle"),
                PipelineEdge(source="middle", target="end"),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Should skip 'middle' because router explicitly targets 'end'
        assert ctx.get("end_done") is True
        assert ctx.get("middle_done") is None

    async def test_failed_node_records_error(self) -> None:
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="fail_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="fail", is_start=True),
                "end": PipelineNode(name="end", handler_type="echo", is_terminal=True),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("_last_error") == "intentional failure"
        assert ctx.get("_failed_node") == "start"

    async def test_missing_handler_raises(self) -> None:
        registry = HandlerRegistry()
        pipeline = _simple_pipeline()

        engine = PipelineEngine(registry=registry)
        with pytest.raises(EngineError, match="No handler registered"):
            await engine.run(pipeline)

    async def test_max_steps_limit(self, registry: HandlerRegistry) -> None:
        """A looping pipeline is capped by max_steps."""
        pipeline = Pipeline(
            name="loop",
            nodes={
                "a": PipelineNode(name="a", handler_type="echo", is_start=True),
                "b": PipelineNode(name="b", handler_type="echo"),
            },
            edges=[
                PipelineEdge(source="a", target="b"),
                PipelineEdge(source="b", target="a"),
            ],
            start_node="a",
        )

        engine = PipelineEngine(registry=registry, max_steps=10)
        ctx = await engine.run(pipeline)
        completed = ctx.get("_completed_nodes")
        assert len(completed) == 10

    async def test_checkpoint_records_next_node_not_completed(
        self, registry: HandlerRegistry, tmp_path
    ) -> None:
        """Regression for A3: checkpoint should record the *next* node,
        not the one that just completed."""
        engine = PipelineEngine(registry=registry, checkpoint_dir=str(tmp_path))
        await engine.run(_simple_pipeline())

        # Find checkpoint files
        import json

        cp_files = sorted(tmp_path.glob("checkpoint_*.json"))
        assert len(cp_files) >= 1

        # Verify the invariant: for every checkpoint, current_node
        # should NOT be in completed_nodes (unless it's the terminal
        # checkpoint where current_node IS the terminal that just ran).
        # At minimum, after "start" completes, the checkpoint should
        # record the next node to run.
        for cpf in cp_files:
            cp_data = json.loads(cpf.read_text())
            completed = cp_data["completed_nodes"]
            current = cp_data["current_node"]
            # If more nodes were executed after this checkpoint was saved,
            # current_node is the NEXT to execute, which hasn't yet been added
            # to completed_nodes.
            # The key property: current_node points forward, not backward.
            if len(completed) < 3:  # not the final checkpoint
                assert current not in completed[: len(completed)]

    async def test_resume_from_checkpoint_skips_completed_nodes(
        self, registry: HandlerRegistry, echo_handler: EchoHandler
    ) -> None:
        from attractor.pipeline.models import Checkpoint

        pipeline = _simple_pipeline()
        # Create a checkpoint that says we're at "middle" with "start" done
        checkpoint = Checkpoint(
            pipeline_name="test",
            current_node="middle",
            context=PipelineContext(),
            completed_nodes=["start"],
        )

        engine = PipelineEngine(registry=registry)
        await engine.run(pipeline, checkpoint=checkpoint)

        # Handler should have been called for middle and end, NOT start
        assert "start" not in echo_handler.executed
        assert "middle" in echo_handler.executed
        assert "end" in echo_handler.executed

    async def test_resume_with_invalid_node_raises(
        self, registry: HandlerRegistry
    ) -> None:
        from attractor.pipeline.models import Checkpoint

        pipeline = _simple_pipeline()
        checkpoint = Checkpoint(
            pipeline_name="test",
            current_node="nonexistent_node",
            context=PipelineContext(),
            completed_nodes=[],
        )

        engine = PipelineEngine(registry=registry)
        with pytest.raises(EngineError, match="not found"):
            await engine.run(pipeline, checkpoint=checkpoint)

    async def test_condition_error_sets_context_key(
        self, registry: HandlerRegistry
    ) -> None:
        """Regression for A2: broken condition → _condition_error in context."""
        pipeline = Pipeline(
            name="cond_error",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "target": PipelineNode(
                    name="target", handler_type="echo", is_terminal=True
                ),
                "fallback": PipelineNode(
                    name="fallback", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="target",
                    condition="invalid!!! syntax @@@",
                    weight=1,
                ),
                PipelineEdge(source="start", target="fallback", weight=0),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # The bad condition should have set _condition_error
        assert ctx.has("_condition_error")
        # But execution continued via fallback
        assert ctx.get("fallback_done") is True

    async def test_condition_error_on_only_edge_raises(
        self, registry: HandlerRegistry
    ) -> None:
        pipeline = Pipeline(
            name="only_bad_edge",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "target": PipelineNode(
                    name="target", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="target",
                    condition="invalid!!! syntax @@@",
                ),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        with pytest.raises(EngineError, match="No edge matched"):
            await engine.run(pipeline)

    async def test_default_edge_when_no_condition_matches(
        self, registry: HandlerRegistry
    ) -> None:
        pipeline = Pipeline(
            name="default_edge",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "cond_target": PipelineNode(
                    name="cond_target", handler_type="echo", is_terminal=True
                ),
                "default_target": PipelineNode(
                    name="default_target", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="cond_target",
                    condition="never_true == true",
                    weight=1,
                ),
                PipelineEdge(
                    source="start", target="default_target", weight=0
                ),  # no condition = default
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("default_target_done") is True
        assert ctx.get("cond_target_done") is None
