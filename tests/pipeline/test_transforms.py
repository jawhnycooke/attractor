"""Tests for the pipeline transform system (P-C08)."""

import pytest

from attractor.pipeline.models import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
)
from attractor.pipeline.transforms import (
    Transform,
    TransformRegistry,
    VariableExpansionTransform,
    create_default_registry,
)


class TestTransformProtocol:
    """Tests for the Transform protocol."""

    def test_custom_class_satisfies_protocol(self) -> None:
        """A class with apply(pipeline) -> Pipeline satisfies Transform."""

        class MyTransform:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        assert isinstance(MyTransform(), Transform)

    def test_class_without_apply_does_not_satisfy(self) -> None:
        """A class without apply does not satisfy Transform."""

        class NotATransform:
            def run(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        assert not isinstance(NotATransform(), Transform)


class TestVariableExpansionTransform:
    """Tests for the built-in variable expansion transform."""

    def test_expands_goal_in_prompt(self) -> None:
        """$goal in node prompts is replaced with the pipeline goal."""
        pipeline = Pipeline(
            name="test",
            goal="fix the login bug",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="Task: $goal"
                ),
            },
        )
        transform = VariableExpansionTransform()
        result = transform.apply(pipeline)
        assert result.nodes["n1"].prompt == "Task: fix the login bug"

    def test_expands_multiple_occurrences(self) -> None:
        """Multiple $goal occurrences in a single prompt are all expanded."""
        pipeline = Pipeline(
            name="test",
            goal="deploy v2",
            nodes={
                "n1": PipelineNode(
                    name="n1",
                    handler_type="codergen",
                    prompt="First: $goal. Then verify: $goal",
                ),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result.nodes["n1"].prompt == "First: deploy v2. Then verify: deploy v2"

    def test_expands_across_multiple_nodes(self) -> None:
        """$goal is expanded in all nodes, not just the first."""
        pipeline = Pipeline(
            name="test",
            goal="refactor auth",
            nodes={
                "a": PipelineNode(
                    name="a", handler_type="codergen", prompt="Step 1: $goal"
                ),
                "b": PipelineNode(
                    name="b", handler_type="codergen", prompt="Step 2: $goal"
                ),
                "c": PipelineNode(
                    name="c", handler_type="codergen", prompt="No variable here"
                ),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result.nodes["a"].prompt == "Step 1: refactor auth"
        assert result.nodes["b"].prompt == "Step 2: refactor auth"
        assert result.nodes["c"].prompt == "No variable here"

    def test_no_goal_leaves_prompts_unchanged(self) -> None:
        """When pipeline has no goal, $goal is NOT expanded."""
        pipeline = Pipeline(
            name="test",
            goal="",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="Task: $goal"
                ),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result.nodes["n1"].prompt == "Task: $goal"

    def test_no_dollar_goal_in_prompt_is_noop(self) -> None:
        """Prompts without $goal are unaffected."""
        pipeline = Pipeline(
            name="test",
            goal="something",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="plain prompt"
                ),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result.nodes["n1"].prompt == "plain prompt"

    def test_empty_prompt_is_noop(self) -> None:
        """Nodes with empty prompts are skipped."""
        pipeline = Pipeline(
            name="test",
            goal="goal value",
            nodes={
                "n1": PipelineNode(name="n1", handler_type="start", prompt=""),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result.nodes["n1"].prompt == ""

    def test_returns_same_pipeline_object(self) -> None:
        """Transform modifies pipeline in place and returns it."""
        pipeline = Pipeline(
            name="test",
            goal="x",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="$goal"
                ),
            },
        )
        result = VariableExpansionTransform().apply(pipeline)
        assert result is pipeline


class TestTransformRegistry:
    """Tests for TransformRegistry ordering and application."""

    def test_empty_registry_returns_pipeline_unchanged(self) -> None:
        """An empty registry returns the pipeline as-is."""
        registry = TransformRegistry()
        pipeline = Pipeline(
            name="test",
            nodes={"n1": PipelineNode(name="n1", handler_type="echo")},
        )
        result = registry.apply_all(pipeline)
        assert result.nodes["n1"].handler_type == "echo"

    def test_builtin_transforms_run_before_custom(self) -> None:
        """Built-in transforms execute before custom transforms."""
        order: list[str] = []

        class BuiltinT:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                order.append("builtin")
                return pipeline

        class CustomT:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                order.append("custom")
                return pipeline

        registry = TransformRegistry()
        registry.register_builtin(BuiltinT())
        registry.register(CustomT())
        registry.apply_all(Pipeline(name="test"))

        assert order == ["builtin", "custom"]

    def test_multiple_custom_transforms_run_in_registration_order(self) -> None:
        """Custom transforms run in the order they were registered."""
        order: list[int] = []

        class OrderedT:
            def __init__(self, idx: int) -> None:
                self._idx = idx

            def apply(self, pipeline: Pipeline) -> Pipeline:
                order.append(self._idx)
                return pipeline

        registry = TransformRegistry()
        registry.register(OrderedT(1))
        registry.register(OrderedT(2))
        registry.register(OrderedT(3))
        registry.apply_all(Pipeline(name="test"))

        assert order == [1, 2, 3]

    def test_transforms_property_returns_all_in_order(self) -> None:
        """The transforms property returns built-in + custom in order."""

        class T1:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        class T2:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        class T3:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        t1, t2, t3 = T1(), T2(), T3()
        registry = TransformRegistry()
        registry.register_builtin(t1)
        registry.register(t2)
        registry.register(t3)

        transforms = registry.transforms
        assert len(transforms) == 3
        assert transforms[0] is t1
        assert transforms[1] is t2
        assert transforms[2] is t3

    def test_transform_chain_composition(self) -> None:
        """Transforms compose: output of one is input to the next."""

        class AppendSuffix:
            def __init__(self, suffix: str) -> None:
                self._suffix = suffix

            def apply(self, pipeline: Pipeline) -> Pipeline:
                for node in pipeline.nodes.values():
                    node.prompt += self._suffix
                return pipeline

        registry = TransformRegistry()
        registry.register(AppendSuffix("_A"))
        registry.register(AppendSuffix("_B"))

        pipeline = Pipeline(
            name="test",
            nodes={
                "n1": PipelineNode(name="n1", handler_type="echo", prompt="start"),
            },
        )
        result = registry.apply_all(pipeline)
        assert result.nodes["n1"].prompt == "start_A_B"

    def test_builtin_can_feed_into_custom(self) -> None:
        """Built-in variable expansion feeds into custom transforms."""

        class UppercasePrompts:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                for node in pipeline.nodes.values():
                    node.prompt = node.prompt.upper()
                return pipeline

        registry = TransformRegistry()
        registry.register_builtin(VariableExpansionTransform())
        registry.register(UppercasePrompts())

        pipeline = Pipeline(
            name="test",
            goal="fix bugs",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="Task: $goal"
                ),
            },
        )
        result = registry.apply_all(pipeline)
        # Variable expansion first, then uppercase
        assert result.nodes["n1"].prompt == "TASK: FIX BUGS"


class TestCreateDefaultRegistry:
    """Tests for the create_default_registry factory."""

    def test_includes_variable_expansion(self) -> None:
        """Default registry includes VariableExpansionTransform."""
        registry = create_default_registry()
        transforms = registry.transforms
        assert len(transforms) >= 1
        assert any(
            isinstance(t, VariableExpansionTransform) for t in transforms
        )

    def test_default_registry_expands_goal(self) -> None:
        """Default registry correctly expands $goal in prompts."""
        registry = create_default_registry()
        pipeline = Pipeline(
            name="test",
            goal="write tests",
            nodes={
                "n1": PipelineNode(
                    name="n1", handler_type="codergen", prompt="Do: $goal"
                ),
            },
        )
        result = registry.apply_all(pipeline)
        assert result.nodes["n1"].prompt == "Do: write tests"

    def test_default_registry_accepts_custom_transforms(self) -> None:
        """Custom transforms can be added to the default registry."""

        class NoopTransform:
            def apply(self, pipeline: Pipeline) -> Pipeline:
                return pipeline

        registry = create_default_registry()
        registry.register(NoopTransform())
        assert len(registry.transforms) == 2  # builtin + custom
