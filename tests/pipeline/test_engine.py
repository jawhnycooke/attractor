"""Tests for the pipeline execution engine."""

import asyncio
import json
from pathlib import Path

import pytest

from attractor.pipeline.engine import EngineError, PipelineEngine, create_stage_dir
from attractor.pipeline.events import PipelineEvent, PipelineEventEmitter, PipelineEventType
from attractor.pipeline.handlers import HandlerRegistry
from attractor.pipeline.models import (
    Checkpoint,
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

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        self.executed.append(node.name)
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            output=f"echo:{node.name}",
            context_updates={f"{node.name}_done": True},
        )


class FailHandler:
    """Test handler that always fails."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(status=OutcomeStatus.FAIL, failure_reason="intentional failure")


class RetryHandler:
    """Test handler that returns RETRY a configurable number of times."""

    def __init__(self, retries_before_success: int = 1) -> None:
        self._retries_before_success = retries_before_success
        self.attempt_count = 0

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        self.attempt_count += 1
        if self.attempt_count <= self._retries_before_success:
            return NodeResult(status=OutcomeStatus.RETRY)
        return NodeResult(status=OutcomeStatus.SUCCESS, output="ok")


class ExceptionHandler:
    """Test handler that raises an exception a configurable number of times."""

    def __init__(self, exceptions_before_success: int = 1) -> None:
        self._exceptions_before_success = exceptions_before_success
        self.attempt_count = 0

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        self.attempt_count += 1
        if self.attempt_count <= self._exceptions_before_success:
            raise RuntimeError("handler crashed")
        return NodeResult(status=OutcomeStatus.SUCCESS, output="recovered")


class SlowHandler:
    """Test handler that sleeps for a configurable duration."""

    def __init__(self, sleep_seconds: float = 5.0) -> None:
        self._sleep_seconds = sleep_seconds

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        await asyncio.sleep(self._sleep_seconds)
        return NodeResult(status=OutcomeStatus.SUCCESS, output="done")


class ContextSetHandler:
    """Test handler that sets a specific context key to a given value."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(
            status=OutcomeStatus.SUCCESS,
            context_updates={self._key: self._value},
        )


class SkippedHandler:
    """Test handler that returns SKIPPED."""

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SKIPPED)


class RoutingHandler:
    """Test handler that routes to a specific next node."""

    def __init__(self, target: str) -> None:
        self._target = target

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(status=OutcomeStatus.SUCCESS, next_node=self._target)


class PreferredLabelHandler:
    """Test handler that returns a preferred_label."""

    def __init__(self, label: str) -> None:
        self._label = label

    async def execute(
        self,
        node: PipelineNode,
        context: PipelineContext,
        graph: Pipeline | None = None,
        logs_root: Path | None = None,
    ) -> NodeResult:
        return NodeResult(
            status=OutcomeStatus.SUCCESS, preferred_label=self._label
        )


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
                source="start", target="left", condition="go_left=true", weight=1
            ),
            PipelineEdge(
                source="start", target="right", condition="go_left=false", weight=0
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

    async def test_failed_node_raises_with_no_failure_route(self) -> None:
        """E7: FAIL with no failure route raises EngineError."""
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
        ctx = PipelineContext()
        with pytest.raises(EngineError, match="failed with no failure route"):
            await engine.run(pipeline, context=ctx)
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

        for cpf in cp_files:
            cp_data = json.loads(cpf.read_text())
            completed = cp_data["completed_nodes"]
            current = cp_data["current_node"]
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
                    condition="x == broken",
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

    async def test_condition_error_on_only_edge_uses_fallback(
        self, registry: HandlerRegistry
    ) -> None:
        """E16: When only edge has a broken condition, fallback picks it."""
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
                    condition="x == broken",
                ),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # E16: Fallback picks any edge
        assert ctx.has("_condition_error")
        assert ctx.get("target_done") is True

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
                    condition="never_true=true",
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


class TestEdgeSelection:
    """Tests for the 5-step edge selection algorithm (E1, E2, E15, E16)."""

    @pytest.fixture
    def registry(self) -> HandlerRegistry:
        reg = HandlerRegistry()
        reg.register("echo", EchoHandler())
        return reg

    async def test_e1_condition_short_circuit(self, registry: HandlerRegistry) -> None:
        """E1: When a conditional edge matches, unconditional edges are skipped."""
        pipeline = Pipeline(
            name="e1_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "cond": PipelineNode(name="cond", handler_type="echo", is_terminal=True),
                "uncond": PipelineNode(
                    name="uncond", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="cond",
                    condition="outcome=success",
                    weight=0,
                ),
                PipelineEdge(source="start", target="uncond", weight=10),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Conditional edge matched → short-circuit; unconditional skipped
        assert ctx.get("cond_done") is True
        assert ctx.get("uncond_done") is None

    async def test_e2_label_normalization_accelerator_prefix(self) -> None:
        """E2: Accelerator prefixes ([Y], Y), Y -) are stripped during label match."""
        registry = HandlerRegistry()
        registry.register("label", PreferredLabelHandler("yes"))
        registry.register("echo", EchoHandler())

        for prefix in ["[Y] ", "Y) ", "Y - "]:
            pipeline = Pipeline(
                name="e2_test",
                nodes={
                    "start": PipelineNode(
                        name="start", handler_type="label", is_start=True
                    ),
                    "yes_node": PipelineNode(
                        name="yes_node", handler_type="echo", is_terminal=True
                    ),
                    "no_node": PipelineNode(
                        name="no_node", handler_type="echo", is_terminal=True
                    ),
                },
                edges=[
                    PipelineEdge(
                        source="start",
                        target="yes_node",
                        label=f"{prefix}Yes",
                    ),
                    PipelineEdge(
                        source="start", target="no_node", label="No"
                    ),
                ],
                start_node="start",
            )

            engine = PipelineEngine(registry=registry)
            ctx = await engine.run(pipeline)
            assert ctx.get("yes_node_done") is True, f"Failed for prefix {prefix!r}"

    async def test_e15_weight_sort_unconditional_only(
        self, registry: HandlerRegistry
    ) -> None:
        """E15: Steps 4/5 sort only unconditional edges by weight."""
        pipeline = Pipeline(
            name="e15_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "heavy_cond": PipelineNode(
                    name="heavy_cond", handler_type="echo", is_terminal=True
                ),
                "light_uncond": PipelineNode(
                    name="light_uncond", handler_type="echo", is_terminal=True
                ),
                "heavy_uncond": PipelineNode(
                    name="heavy_uncond", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="heavy_cond",
                    condition="never_true=true",
                    weight=100,
                ),
                PipelineEdge(source="start", target="light_uncond", weight=1),
                PipelineEdge(source="start", target="heavy_uncond", weight=5),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # heavy_cond is conditional and doesn't match → excluded from step 4/5
        # heavy_uncond has higher weight than light_uncond → selected
        assert ctx.get("heavy_uncond_done") is True

    async def test_e15_lexical_tiebreak(self, registry: HandlerRegistry) -> None:
        """E15: Equal-weight unconditional edges tiebreak lexically by target."""
        pipeline = Pipeline(
            name="e15_lex",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "beta": PipelineNode(
                    name="beta", handler_type="echo", is_terminal=True
                ),
                "alpha": PipelineNode(
                    name="alpha", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(source="start", target="beta", weight=0),
                PipelineEdge(source="start", target="alpha", weight=0),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("alpha_done") is True
        assert ctx.get("beta_done") is None

    async def test_e16_fallback_any_edge(self, registry: HandlerRegistry) -> None:
        """E16: When no unconditional edge exists, fallback uses any edge."""
        pipeline = Pipeline(
            name="e16_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "target_a": PipelineNode(
                    name="target_a", handler_type="echo", is_terminal=True
                ),
                "target_b": PipelineNode(
                    name="target_b", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="target_a",
                    condition="never_match=true",
                    weight=5,
                ),
                PipelineEdge(
                    source="start",
                    target="target_b",
                    condition="also_never=true",
                    weight=10,
                ),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Both conditions fail → fallback picks best by weight → target_b
        assert ctx.get("target_b_done") is True


class TestRetryLogic:
    """Tests for retry, backoff, and exception handling (E4, E5, E6)."""

    async def test_e4_inherit_max_retries_from_pipeline(self) -> None:
        """E4: When node.max_retries is 0, inherit from pipeline metadata."""
        retry_handler = RetryHandler(retries_before_success=2)
        registry = HandlerRegistry()
        registry.register("retry", retry_handler)
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e4_test",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="retry", is_start=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
            metadata={"default_max_retry": 3},
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Should succeed after 3 attempts (2 retries + 1 success)
        assert retry_handler.attempt_count == 3
        assert ctx.get("end_done") is True

    async def test_e5_allow_partial_on_exhausted_retries(self) -> None:
        """E5: allow_partial=true returns PARTIAL_SUCCESS when retries exhausted."""
        # Handler always returns RETRY
        retry_handler = RetryHandler(retries_before_success=999)
        registry = HandlerRegistry()
        registry.register("retry", retry_handler)
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e5_partial",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="retry",
                    is_start=True,
                    max_retries=2,
                    allow_partial=True,
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Retries exhausted → PARTIAL_SUCCESS due to allow_partial
        # Pipeline continued to "end" (wouldn't happen if result was FAIL)
        assert retry_handler.attempt_count == 3  # 1 initial + 2 retries
        assert ctx.get("end_done") is True
        # "start" should be in completed (PARTIAL_SUCCESS counts)
        assert "start" in ctx.get("_completed_nodes")

    async def test_e5_fail_on_exhausted_retries_no_allow_partial(self) -> None:
        """E5: Without allow_partial, exhausted retries produce FAIL."""
        retry_handler = RetryHandler(retries_before_success=999)
        registry = HandlerRegistry()
        registry.register("retry", retry_handler)

        pipeline = Pipeline(
            name="e5_fail",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="retry",
                    is_start=True,
                    max_retries=1,
                    retry_target="start",
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        # FAIL with retry_target pointing to self — will loop until max_steps
        # Instead, let's just check the error
        pipeline2 = Pipeline(
            name="e5_fail",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="retry",
                    is_start=True,
                    max_retries=1,
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )
        registry.register("echo", EchoHandler())

        engine = PipelineEngine(registry=registry)
        ctx = PipelineContext()
        with pytest.raises(EngineError, match="failed with no failure route"):
            await engine.run(pipeline2, context=ctx)
        assert ctx.get("outcome") == "fail"

    async def test_e5_catch_handler_exceptions(self) -> None:
        """E5: Handler exceptions are caught and retried."""
        exc_handler = ExceptionHandler(exceptions_before_success=1)
        registry = HandlerRegistry()
        registry.register("exc", exc_handler)
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e5_exc",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="exc",
                    is_start=True,
                    max_retries=2,
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert exc_handler.attempt_count == 2  # 1 exception + 1 success
        assert ctx.get("end_done") is True

    async def test_e5_exception_exhausted_produces_fail(self) -> None:
        """E5: When exceptions exhaust all retries, result is FAIL."""
        exc_handler = ExceptionHandler(exceptions_before_success=999)
        registry = HandlerRegistry()
        registry.register("exc", exc_handler)

        pipeline = Pipeline(
            name="e5_exc_fail",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="exc",
                    is_start=True,
                    max_retries=1,
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = PipelineContext()
        with pytest.raises(EngineError, match="failed with no failure route"):
            await engine.run(pipeline, context=ctx)
        assert ctx.get("_last_error") == "handler crashed"

    async def test_e6_retry_delay_parameters(self) -> None:
        """E6: max_delay=60s, jitter range [0.5x, 1.5x]."""
        engine = PipelineEngine()
        node = PipelineNode(name="test", handler_type="echo")

        delays = [engine._retry_delay(1, node) for _ in range(100)]
        # Base delay for attempt 1 is 0.2s, jitter [0.5, 1.5] → [0.1, 0.3]
        assert all(0.05 <= d <= 0.35 for d in delays)

        # High attempt should cap at max_delay=60s
        high_delays = [engine._retry_delay(20, node) for _ in range(100)]
        assert all(d <= 60 * 1.5 + 0.01 for d in high_delays)
        assert all(d >= 60 * 0.5 - 0.01 for d in high_delays)


class TestFailureRouting:
    """Tests for failure routing (E7, E13)."""

    async def test_e7_fail_edge_routing(self) -> None:
        """E7: FAIL routes through edge with outcome=fail condition."""
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e7_fail_edge",
            nodes={
                "start": PipelineNode(name="start", handler_type="fail", is_start=True),
                "recovery": PipelineNode(
                    name="recovery", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="start",
                    target="recovery",
                    condition="outcome=fail",
                ),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("recovery_done") is True
        assert ctx.get("_last_error") == "intentional failure"

    async def test_e7_retry_target_routing(self) -> None:
        """E7: FAIL routes to node retry_target when no fail edge."""
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e7_retry",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="fail",
                    is_start=True,
                    retry_target="recovery",
                ),
                "recovery": PipelineNode(
                    name="recovery", handler_type="echo", is_terminal=True
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("recovery_done") is True

    async def test_e7_fallback_retry_target_routing(self) -> None:
        """E7: FAIL routes to node fallback_retry_target as last resort."""
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e7_fallback",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="fail",
                    is_start=True,
                    fallback_retry_target="recovery",
                ),
                "recovery": PipelineNode(
                    name="recovery", handler_type="echo", is_terminal=True
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("recovery_done") is True

    async def test_e13_fail_no_route_raises(self) -> None:
        """E13: Non-terminal FAIL with no fail edge and no retry target raises."""
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())

        pipeline = Pipeline(
            name="e13_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="fail", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="fail", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        with pytest.raises(EngineError, match="failed with no failure route"):
            await engine.run(pipeline)


class TestGoalGates:
    """Tests for goal gate enforcement (E3, E9, E10, E11)."""

    async def test_e3_goal_gate_checked_before_terminal_handler(self) -> None:
        """E3: Goal gate is checked before the terminal handler executes."""
        terminal_handler = EchoHandler()
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())
        registry.register("terminal", terminal_handler)

        pipeline = Pipeline(
            name="e3_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "gate": PipelineNode(
                    name="gate", handler_type="echo", goal_gate=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="terminal", is_terminal=True
                ),
            },
            edges=[
                # Skip gate and go directly to end
                PipelineEdge(source="start", target="end"),
            ],
            start_node="start",
            retry_target="gate",
        )

        engine = PipelineEngine(registry=registry, max_steps=5)
        ctx = await engine.run(pipeline)
        completed = ctx.get("_completed_nodes", [])
        # Engine must redirect to "gate" before allowing terminal "end"
        assert "gate" in completed
        # "end" should NOT have been dispatched — the pipeline ends at "gate"
        # (implicitly terminal with no outgoing edges) after the redirect
        assert not terminal_handler.executed

    async def test_e9_goal_gate_checks_outcome_status(self) -> None:
        """E9: Goal gate checks outcome is SUCCESS/PARTIAL_SUCCESS, not just completed."""
        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e9_test",
            nodes={
                "gate_node": PipelineNode(
                    name="gate_node",
                    handler_type="fail",
                    is_start=True,
                    goal_gate=True,
                    retry_target="gate_node",
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(
                    source="gate_node",
                    target="end",
                    condition="outcome=fail",
                ),
            ],
            start_node="gate_node",
        )

        engine = PipelineEngine(registry=registry, max_steps=5)
        ctx = await engine.run(pipeline)
        # Gate node completed but with FAIL outcome → gate unsatisfied
        # Engine redirects to retry_target (gate_node) each time until max_steps
        assert ctx.has("_goal_gate_unmet")

    async def test_e10_retry_target_order(self) -> None:
        """E10: Retry target resolution: node → node fallback → pipeline → pipeline fallback."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        # Test node-level retry_target
        pipeline = Pipeline(
            name="e10_node",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "gate": PipelineNode(
                    name="gate",
                    handler_type="echo",
                    goal_gate=True,
                    retry_target="gate",
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(source="start", target="end"),
                PipelineEdge(source="gate", target="end"),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, max_steps=5)
        ctx = await engine.run(pipeline)
        # Gate not visited → unsatisfied → redirects to gate via retry_target
        assert "gate" in ctx.get("_completed_nodes", [])

    async def test_e10_pipeline_level_retry_target(self) -> None:
        """E10: Falls back to pipeline-level retry_target."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e10_pipeline",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "gate": PipelineNode(
                    name="gate", handler_type="echo", goal_gate=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(source="start", target="end"),
                PipelineEdge(source="gate", target="end"),
            ],
            start_node="start",
            retry_target="gate",
        )

        engine = PipelineEngine(registry=registry, max_steps=5)
        ctx = await engine.run(pipeline)
        assert "gate" in ctx.get("_completed_nodes", [])

    async def test_e11_no_retry_target_raises(self) -> None:
        """E11: Unsatisfied goal gate with no retry target raises EngineError."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e11_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "gate": PipelineNode(
                    name="gate", handler_type="echo", goal_gate=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        with pytest.raises(EngineError, match="Goal gate.*unsatisfied"):
            await engine.run(pipeline)


class TestMiscFeatures:
    """Tests for loop_restart and SKIPPED status (E12, E14)."""

    async def test_e12_loop_restart_resets_state(self) -> None:
        """E12: Edge with loop_restart=true resets completed nodes."""
        echo = EchoHandler()
        registry = HandlerRegistry()
        registry.register("echo", echo)

        pipeline = Pipeline(
            name="e12_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "middle": PipelineNode(name="middle", handler_type="echo"),
                "end": PipelineNode(name="end", handler_type="echo", is_terminal=True),
            },
            edges=[
                PipelineEdge(source="start", target="middle"),
                PipelineEdge(
                    source="middle",
                    target="start",
                    condition="outcome=success",
                    loop_restart=True,
                    weight=1,
                ),
                PipelineEdge(source="middle", target="end", weight=0),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, max_steps=10)
        ctx = await engine.run(pipeline)
        completed = ctx.get("_completed_nodes")
        # Without loop_restart, all 10 steps would accumulate in completed.
        # With loop_restart, completed resets each cycle (every 2 steps),
        # so only the tail of the last partial iteration remains.
        assert len(completed) <= 2

    async def test_e14_skipped_not_recorded(self) -> None:
        """E14: SKIPPED status skips outcome recording."""
        registry = HandlerRegistry()
        registry.register("skip", SkippedHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="e14_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="skip", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        completed = ctx.get("_completed_nodes")
        # "start" was SKIPPED so not in completed
        assert "start" not in completed
        assert "end" in completed


class TestTimeoutEnforcement:
    """Tests for per-node timeout enforcement."""

    async def test_node_timeout_triggers_failure(self) -> None:
        """A handler that exceeds the node timeout produces a FAIL result."""
        registry = HandlerRegistry()
        registry.register("slow", SlowHandler(sleep_seconds=5.0))

        pipeline = Pipeline(
            name="timeout_test",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="slow",
                    is_start=True,
                    timeout=0.1,
                    retry_target="recovery",
                ),
                "recovery": PipelineNode(
                    name="recovery",
                    handler_type="slow",
                    is_terminal=True,
                    timeout=0.1,
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = PipelineContext()
        with pytest.raises(EngineError, match="failed with no failure route"):
            await engine.run(pipeline, context=ctx)
        assert "timed out" in ctx.get("_last_error", "")
        assert ctx.get("_failed_node") is not None

    async def test_node_without_timeout_runs_normally(self) -> None:
        """A handler completes normally when no timeout is set."""
        registry = HandlerRegistry()
        registry.register("slow", SlowHandler(sleep_seconds=0.01))
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="no_timeout",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="slow", is_start=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("end_done") is True


class TestContextCloneIsolation:
    """Tests for PipelineContext.clone() and create_scope() isolation."""

    async def test_clone_returns_deep_copy(self) -> None:
        """clone() returns a deep copy — mutations don't affect the original."""
        original = PipelineContext.from_dict(
            {"key": "original", "nested": {"a": 1}}
        )
        cloned = original.clone()

        # Mutate the clone
        cloned.set("key", "modified")
        cloned.get("nested")["a"] = 999

        # Original is unaffected
        assert original.get("key") == "original"
        assert original.get("nested")["a"] == 1

    async def test_create_scope_returns_clone_not_empty(self) -> None:
        """create_scope() returns a clone of the parent, not an empty context."""
        parent = PipelineContext.from_dict({"shared": "value", "count": 42})
        scope = parent.create_scope("branch")

        assert scope.get("shared") == "value"
        assert scope.get("count") == 42

        # Mutate scope — parent unaffected
        scope.set("shared", "branch_value")
        assert parent.get("shared") == "value"

    async def test_parallel_branches_isolated(self) -> None:
        """Simulated parallel branches get isolated context copies."""
        registry = HandlerRegistry()
        registry.register("set_a", ContextSetHandler("x", "A"))
        registry.register("set_b", ContextSetHandler("x", "B"))
        registry.register("echo", EchoHandler())

        # Simulate what a parallel handler would do:
        # clone parent context for each branch, run handlers, verify isolation
        parent_ctx = PipelineContext.from_dict({"x": "parent"})

        branch_a_ctx = parent_ctx.clone()
        branch_b_ctx = parent_ctx.clone()

        node_a = PipelineNode(name="branch_a", handler_type="set_a")
        node_b = PipelineNode(name="branch_b", handler_type="set_b")

        handler_a = registry.get("set_a")
        handler_b = registry.get("set_b")

        result_a = await handler_a.execute(node_a, branch_a_ctx)
        result_b = await handler_b.execute(node_b, branch_b_ctx)

        # Apply context updates to each branch
        branch_a_ctx.update(result_a.context_updates)
        branch_b_ctx.update(result_b.context_updates)

        # Each branch has its own value
        assert branch_a_ctx.get("x") == "A"
        assert branch_b_ctx.get("x") == "B"

        # Parent context is unchanged
        assert parent_ctx.get("x") == "parent"


class TestCheckpointNodeRetries:
    """Tests for S3-ckpt: node_retries populated in checkpoints."""

    async def test_checkpoint_contains_node_retries(self, tmp_path: Path) -> None:
        """S3-ckpt: node_retries extracted from context retry keys."""
        retry_handler = RetryHandler(retries_before_success=2)
        registry = HandlerRegistry()
        registry.register("retry", retry_handler)
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="ckpt_retries",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="retry", is_start=True, max_retries=3
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, checkpoint_dir=str(tmp_path))
        await engine.run(pipeline)

        cp_files = sorted(tmp_path.glob("checkpoint_*.json"))
        assert len(cp_files) >= 1

        # Check the last checkpoint
        last_cp = json.loads(cp_files[-1].read_text())
        assert "node_retries" in last_cp
        # "start" had 2 retries before success → retry_count = 2
        assert last_cp["node_retries"]["start"] == 2

    async def test_checkpoint_retries_zero_for_no_retry(self, tmp_path: Path) -> None:
        """Nodes without retries get retry_count=0 in checkpoint."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="no_retry",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, checkpoint_dir=str(tmp_path))
        await engine.run(pipeline)

        cp_files = sorted(tmp_path.glob("checkpoint_*.json"))
        assert len(cp_files) >= 1

        last_cp = json.loads(cp_files[-1].read_text())
        # Nodes completed on first attempt have retry_count = 0
        assert last_cp["node_retries"].get("start") == 0
        assert last_cp["node_retries"].get("end") == 0


class TestCreateStageDir:
    """Tests for #13: create_stage_dir utility."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        stage_dir = create_stage_dir(tmp_path, "my_node")
        assert stage_dir.exists()
        assert stage_dir.is_dir()
        assert stage_dir == tmp_path / "my_node"

    def test_creates_nested_directory(self, tmp_path: Path) -> None:
        logs_root = tmp_path / "runs" / "2026-01-01"
        stage_dir = create_stage_dir(logs_root, "build")
        assert stage_dir.exists()
        assert stage_dir == logs_root / "build"

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling twice doesn't raise."""
        create_stage_dir(tmp_path, "node1")
        create_stage_dir(tmp_path, "node1")
        assert (tmp_path / "node1").exists()

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        result = create_stage_dir(tmp_path, "code_review")
        assert result.name == "code_review"
        assert result.parent == tmp_path


class TestFidelityModes:
    """Tests for #14: context fidelity mode tracking."""

    async def test_fidelity_mode_recorded_in_context(self) -> None:
        """When a node has fidelity set, _fidelity_mode is set in context."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="fidelity_test",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="echo",
                    is_start=True,
                    fidelity="compact",
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("_fidelity_mode") == "compact"

    async def test_fidelity_mode_changes_with_nodes(self) -> None:
        """Each node with fidelity updates _fidelity_mode."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="fidelity_multi",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="echo",
                    is_start=True,
                    fidelity="full",
                ),
                "middle": PipelineNode(
                    name="middle",
                    handler_type="echo",
                    fidelity="truncate",
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[
                PipelineEdge(source="start", target="middle"),
                PipelineEdge(source="middle", target="end"),
            ],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        # Last node with fidelity was "middle" with "truncate"
        assert ctx.get("_fidelity_mode") == "truncate"

    async def test_no_fidelity_does_not_set_key(self) -> None:
        """Nodes without fidelity don't set _fidelity_mode."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="no_fidelity",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="echo", is_start=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert not ctx.has("_fidelity_mode")

    def test_apply_fidelity_sets_context(self) -> None:
        """_apply_fidelity static method sets the context key."""
        ctx = PipelineContext()
        PipelineEngine._apply_fidelity(ctx, "summary:high")
        assert ctx.get("_fidelity_mode") == "summary:high"


class TestPipelineEvents:
    """Tests for #15: pipeline event system."""

    async def test_event_emitter_fires_events(self) -> None:
        """PipelineEventEmitter calls registered callbacks."""
        received: list[PipelineEvent] = []

        async def on_event(event: PipelineEvent) -> None:
            received.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_START, on_event)

        event = PipelineEvent(
            type=PipelineEventType.NODE_START,
            node_name="test",
            pipeline_name="pipe",
        )
        await emitter.emit(event)

        assert len(received) == 1
        assert received[0].node_name == "test"

    async def test_event_emitter_multiple_listeners(self) -> None:
        """Multiple listeners for the same event type are all called."""
        calls: list[str] = []

        async def listener_a(event: PipelineEvent) -> None:
            calls.append("a")

        async def listener_b(event: PipelineEvent) -> None:
            calls.append("b")

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_COMPLETE, listener_a)
        emitter.on(PipelineEventType.NODE_COMPLETE, listener_b)

        await emitter.emit(PipelineEvent(type=PipelineEventType.NODE_COMPLETE))
        assert calls == ["a", "b"]

    async def test_event_emitter_different_types_isolated(self) -> None:
        """Listeners only receive events of their registered type."""
        received_starts: list[PipelineEvent] = []
        received_completes: list[PipelineEvent] = []

        async def on_start(event: PipelineEvent) -> None:
            received_starts.append(event)

        async def on_complete(event: PipelineEvent) -> None:
            received_completes.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_START, on_start)
        emitter.on(PipelineEventType.NODE_COMPLETE, on_complete)

        await emitter.emit(PipelineEvent(type=PipelineEventType.NODE_START))
        assert len(received_starts) == 1
        assert len(received_completes) == 0

    async def test_event_emitter_callback_error_does_not_block_others(self) -> None:
        """An error in one callback does not prevent others from firing."""
        calls: list[str] = []

        async def failing(event: PipelineEvent) -> None:
            raise RuntimeError("callback error")

        async def succeeding(event: PipelineEvent) -> None:
            calls.append("ok")

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_START, failing)
        emitter.on(PipelineEventType.NODE_START, succeeding)

        await emitter.emit(PipelineEvent(type=PipelineEventType.NODE_START))
        assert calls == ["ok"]

    async def test_event_emitter_listeners_property(self) -> None:
        """The listeners property returns the mapping."""
        emitter = PipelineEventEmitter()

        async def noop(event: PipelineEvent) -> None:
            pass

        emitter.on(PipelineEventType.PIPELINE_START, noop)
        listeners = emitter.listeners
        assert PipelineEventType.PIPELINE_START in listeners
        assert len(listeners[PipelineEventType.PIPELINE_START]) == 1

    async def test_engine_emits_pipeline_start_and_complete(self) -> None:
        """Engine emits PIPELINE_START and PIPELINE_COMPLETE events."""
        received: list[PipelineEvent] = []

        async def recorder(event: PipelineEvent) -> None:
            received.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.PIPELINE_START, recorder)
        emitter.on(PipelineEventType.PIPELINE_COMPLETE, recorder)

        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="event_test",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, event_emitter=emitter)
        await engine.run(pipeline)

        event_types = [e.type for e in received]
        assert PipelineEventType.PIPELINE_START in event_types
        assert PipelineEventType.PIPELINE_COMPLETE in event_types
        assert received[0].pipeline_name == "event_test"

    async def test_engine_emits_node_start_and_complete(self) -> None:
        """Engine emits NODE_START and NODE_COMPLETE for each node."""
        received: list[PipelineEvent] = []

        async def recorder(event: PipelineEvent) -> None:
            received.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_START, recorder)
        emitter.on(PipelineEventType.NODE_COMPLETE, recorder)

        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="node_events",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, event_emitter=emitter)
        await engine.run(pipeline)

        node_starts = [e for e in received if e.type == PipelineEventType.NODE_START]
        node_completes = [e for e in received if e.type == PipelineEventType.NODE_COMPLETE]

        assert len(node_starts) == 2  # start + end
        assert len(node_completes) == 2
        assert node_starts[0].node_name == "start"
        assert node_starts[1].node_name == "end"

    async def test_engine_emits_node_fail_event(self) -> None:
        """Engine emits NODE_FAIL when a handler fails."""
        received: list[PipelineEvent] = []

        async def recorder(event: PipelineEvent) -> None:
            received.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_FAIL, recorder)

        registry = HandlerRegistry()
        registry.register("fail", FailHandler())
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="fail_event",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="fail",
                    is_start=True,
                    retry_target="recovery",
                ),
                "recovery": PipelineNode(
                    name="recovery", handler_type="echo", is_terminal=True
                ),
            },
            edges=[],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, event_emitter=emitter)
        await engine.run(pipeline)

        assert len(received) == 1
        assert received[0].type == PipelineEventType.NODE_FAIL
        assert received[0].node_name == "start"
        assert received[0].data["failure_reason"] == "intentional failure"

    async def test_engine_emits_node_retry_event(self) -> None:
        """Engine emits NODE_RETRY on retry attempts."""
        received: list[PipelineEvent] = []

        async def recorder(event: PipelineEvent) -> None:
            received.append(event)

        emitter = PipelineEventEmitter()
        emitter.on(PipelineEventType.NODE_RETRY, recorder)

        retry_handler = RetryHandler(retries_before_success=1)
        registry = HandlerRegistry()
        registry.register("retry", retry_handler)
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="retry_event",
            nodes={
                "start": PipelineNode(
                    name="start",
                    handler_type="retry",
                    is_start=True,
                    max_retries=2,
                ),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, event_emitter=emitter)
        await engine.run(pipeline)

        assert len(received) == 1
        assert received[0].type == PipelineEventType.NODE_RETRY
        assert received[0].node_name == "start"
        assert received[0].data["attempt"] == 2

    async def test_engine_no_emitter_doesnt_error(self) -> None:
        """Engine runs without an event emitter."""
        registry = HandlerRegistry()
        registry.register("echo", EchoHandler())

        pipeline = Pipeline(
            name="no_emitter",
            nodes={
                "start": PipelineNode(name="start", handler_type="echo", is_start=True),
                "end": PipelineNode(
                    name="end", handler_type="echo", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)
        assert ctx.get("end_done") is True

    async def test_pipeline_event_type_values(self) -> None:
        """PipelineEventType enum has expected values."""
        assert PipelineEventType.PIPELINE_START == "pipeline_start"
        assert PipelineEventType.PIPELINE_COMPLETE == "pipeline_complete"
        assert PipelineEventType.PIPELINE_FAILED == "pipeline_failed"
        assert PipelineEventType.NODE_START == "node_start"
        assert PipelineEventType.NODE_COMPLETE == "node_complete"
        assert PipelineEventType.NODE_RETRY == "node_retry"
        assert PipelineEventType.NODE_FAIL == "node_fail"
        assert PipelineEventType.CHECKPOINT_SAVED == "checkpoint_saved"

    async def test_pipeline_event_defaults(self) -> None:
        """PipelineEvent has sensible defaults."""
        event = PipelineEvent(type=PipelineEventType.PIPELINE_START)
        assert event.node_name == ""
        assert event.pipeline_name == ""
        assert isinstance(event.timestamp, float)
        assert event.data == {}


class TestLogsRootPassthrough:
    """Tests for P-C12: engine passes logs_root to handlers."""

    async def test_engine_passes_logs_root_to_handler(self, tmp_path: Path) -> None:
        """Engine with logs_root passes it through to handler.execute()."""
        received_logs_root: list[Path | None] = []

        class CapturingHandler:
            async def execute(self, node, context, graph=None, logs_root=None):
                received_logs_root.append(logs_root)
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    context_updates={f"{node.name}_done": True},
                )

        registry = HandlerRegistry()
        registry.register("capture", CapturingHandler())

        pipeline = Pipeline(
            name="logs_root_test",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="capture", is_start=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="capture", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry, logs_root=tmp_path)
        ctx = await engine.run(pipeline)

        assert ctx.get("start_done") is True
        assert ctx.get("end_done") is True
        # Both handlers should have received the logs_root path
        assert len(received_logs_root) == 2
        assert all(lr == tmp_path for lr in received_logs_root)

    async def test_engine_without_logs_root_passes_none(self) -> None:
        """Engine without logs_root passes None to handler.execute()."""
        received_logs_root: list[Path | None] = []

        class CapturingHandler:
            async def execute(self, node, context, graph=None, logs_root=None):
                received_logs_root.append(logs_root)
                return NodeResult(
                    status=OutcomeStatus.SUCCESS,
                    context_updates={f"{node.name}_done": True},
                )

        registry = HandlerRegistry()
        registry.register("capture", CapturingHandler())

        pipeline = Pipeline(
            name="no_logs_root",
            nodes={
                "start": PipelineNode(
                    name="start", handler_type="capture", is_start=True
                ),
                "end": PipelineNode(
                    name="end", handler_type="capture", is_terminal=True
                ),
            },
            edges=[PipelineEdge(source="start", target="end")],
            start_node="start",
        )

        engine = PipelineEngine(registry=registry)
        ctx = await engine.run(pipeline)

        assert ctx.get("end_done") is True
        assert len(received_logs_root) == 2
        assert all(lr is None for lr in received_logs_root)
