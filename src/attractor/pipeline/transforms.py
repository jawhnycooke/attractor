"""Pipeline transform system per spec section 9.

Transforms modify the pipeline graph after parsing and before validation.
They enable preprocessing, optimization, and structural rewriting without
modifying the original DOT file.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from attractor.pipeline.models import Pipeline

logger = logging.getLogger(__name__)


@runtime_checkable
class Transform(Protocol):
    """Protocol for pipeline graph transforms.

    Transforms receive a pipeline graph and return a new or modified graph.
    Implementations should avoid mutating the input if possible, but
    in-place mutation is acceptable when documented.
    """

    def apply(self, pipeline: Pipeline) -> Pipeline:
        """Apply this transform to the pipeline graph.

        Args:
            pipeline: The pipeline to transform.

        Returns:
            A new or modified pipeline.
        """
        ...


class VariableExpansionTransform:
    """Expands ``$goal`` in node prompt attributes to the graph-level goal.

    Per spec section 9.2, this replaces occurrences of ``$goal`` in each
    node's ``prompt`` field with the pipeline's ``goal`` attribute value.
    """

    def apply(self, pipeline: Pipeline) -> Pipeline:
        """Replace ``$goal`` in all node prompts with the pipeline goal.

        Args:
            pipeline: The pipeline to transform.

        Returns:
            The pipeline with expanded prompts (modified in place).
        """
        goal = pipeline.goal
        if not goal:
            return pipeline
        for node in pipeline.nodes.values():
            if node.prompt and "$goal" in node.prompt:
                node.prompt = node.prompt.replace("$goal", goal)
        return pipeline


class TransformRegistry:
    """Registry for managing and applying pipeline transforms.

    Built-in transforms run first, followed by custom transforms
    registered via :meth:`register`. Custom transforms run in
    registration order per spec section 9.3.
    """

    def __init__(self) -> None:
        self._builtin: list[Transform] = []
        self._custom: list[Transform] = []

    def register_builtin(self, transform: Transform) -> None:
        """Register a built-in transform (runs before custom transforms).

        Args:
            transform: The transform to register.
        """
        self._builtin.append(transform)

    def register(self, transform: Transform) -> None:
        """Register a custom transform (runs after built-in transforms).

        Args:
            transform: The transform to register.
        """
        self._custom.append(transform)

    def apply_all(self, pipeline: Pipeline) -> Pipeline:
        """Apply all transforms in order: built-in first, then custom.

        Args:
            pipeline: The pipeline to transform.

        Returns:
            The transformed pipeline.
        """
        for transform in self._builtin:
            logger.debug("Applying built-in transform: %s", type(transform).__name__)
            pipeline = transform.apply(pipeline)
        for transform in self._custom:
            logger.debug("Applying custom transform: %s", type(transform).__name__)
            pipeline = transform.apply(pipeline)
        return pipeline

    @property
    def transforms(self) -> list[Transform]:
        """Return all transforms in application order."""
        return list(self._builtin) + list(self._custom)


def create_default_registry() -> TransformRegistry:
    """Create a registry with default built-in transforms.

    Includes the variable expansion transform per spec section 9.2.

    Returns:
        A new :class:`TransformRegistry` with built-in transforms.
    """
    registry = TransformRegistry()
    registry.register_builtin(VariableExpansionTransform())
    return registry
