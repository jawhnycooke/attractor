"""Model stylesheet for pipeline node attribute defaults.

A :class:`ModelStylesheet` holds a list of :class:`StyleRule` entries.
Each rule matches nodes by handler type, name glob, or custom attribute
predicates, and provides default values for model, temperature,
max_tokens, timeout, and retry count.

Node-specific attributes always override stylesheet defaults.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

from attractor.pipeline.models import PipelineNode


@dataclass
class StyleRule:
    """A single rule that may apply defaults to matching nodes.

    Matching criteria (all specified criteria must match):
        - ``handler_type``: exact match on handler_type
        - ``name_pattern``: glob pattern matched against node name
        - ``match_attributes``: dict of key=value that must all be
          present in the node's attributes

    Defaults applied:
        - ``model``, ``temperature``, ``max_tokens``, ``timeout``,
          ``retry_count``
    """

    handler_type: str | None = None
    name_pattern: str | None = None
    match_attributes: dict[str, Any] = field(default_factory=dict)

    # Defaults to apply
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    retry_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def matches(self, node: PipelineNode) -> bool:
        """Return ``True`` if this rule applies to *node*."""
        if self.handler_type is not None and node.handler_type != self.handler_type:
            return False
        if self.name_pattern is not None and not fnmatch.fnmatch(
            node.name, self.name_pattern
        ):
            return False
        for key, value in self.match_attributes.items():
            if node.attributes.get(key) != value:
                return False
        return True

    def defaults(self) -> dict[str, Any]:
        """Return the non-``None`` default values from this rule."""
        result: dict[str, Any] = {}
        for attr in ("model", "temperature", "max_tokens", "timeout", "retry_count"):
            value = getattr(self, attr)
            if value is not None:
                result[attr] = value
        result.update(self.extra)
        return result


@dataclass
class ModelStylesheet:
    """Ordered collection of :class:`StyleRule` entries.

    Rules are evaluated top-to-bottom; later rules override earlier
    ones for conflicting keys.  Node-specific attributes always win.
    """

    rules: list[StyleRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelStylesheet:
        """Build a stylesheet from a dict (e.g. parsed YAML).

        Expected format::

            {"rules": [
                {"handler_type": "codergen", "model": "gpt-4o", ...},
                {"name_pattern": "test_*", "timeout": 120},
            ]}
        """
        rules: list[StyleRule] = []
        for entry in data.get("rules", []):
            rules.append(
                StyleRule(
                    handler_type=entry.get("handler_type"),
                    name_pattern=entry.get("name_pattern"),
                    match_attributes=entry.get("match_attributes", {}),
                    model=entry.get("model"),
                    temperature=entry.get("temperature"),
                    max_tokens=entry.get("max_tokens"),
                    timeout=entry.get("timeout"),
                    retry_count=entry.get("retry_count"),
                    extra={
                        k: v
                        for k, v in entry.items()
                        if k
                        not in {
                            "handler_type",
                            "name_pattern",
                            "match_attributes",
                            "model",
                            "temperature",
                            "max_tokens",
                            "timeout",
                            "retry_count",
                        }
                    },
                )
            )
        return cls(rules=rules)


def apply_stylesheet(stylesheet: ModelStylesheet, node: PipelineNode) -> dict[str, Any]:
    """Resolve final attributes for *node* by layering stylesheet defaults.

    Evaluation order:
    1. Iterate rules top-to-bottom; collect matching defaults.
    2. Later rules override earlier ones for the same key.
    3. Node's own attributes override everything.

    Returns:
        Merged attribute dict ready for handler consumption.
    """
    resolved: dict[str, Any] = {}
    for rule in stylesheet.rules:
        if rule.matches(node):
            resolved.update(rule.defaults())
    # Node-specific attributes always win
    resolved.update(node.attributes)
    return resolved
