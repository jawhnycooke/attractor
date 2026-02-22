"""Model stylesheet for pipeline node attribute defaults.

A :class:`ModelStylesheet` holds a list of :class:`StyleRule` entries.
Each rule matches nodes by handler type, name glob, CSS-like selectors,
or custom attribute predicates, and provides default values for model,
temperature, max_tokens, timeout, and retry count.

Node-specific attributes always override stylesheet defaults.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from attractor.pipeline.models import Pipeline, PipelineNode


@dataclass
class StyleRule:
    """A single rule that may apply defaults to matching nodes.

    Matching criteria (all specified criteria must match):
        - CSS-style selectors: ``*`` (universal), ``#id``, ``.class``
        - ``handler_type``: exact match on handler_type (legacy)
        - ``name_pattern``: glob pattern matched against node name (legacy)
        - ``match_attributes``: dict of key=value that must all be
          present in the node's attributes (legacy)

    Defaults applied:
        - ``llm_model``, ``llm_provider``, ``reasoning_effort``,
          ``model``, ``temperature``, ``max_tokens``, ``timeout``,
          ``retry_count``
    """

    # CSS-style selector fields
    selector_type: str = "universal"  # "universal", "class", "id"
    selector_value: str = "*"
    specificity: int = 0

    # Legacy selector fields (backward compat)
    handler_type: str | None = None
    name_pattern: str | None = None
    match_attributes: dict[str, Any] = field(default_factory=dict)

    # Spec property names
    llm_model: str | None = None
    llm_provider: str | None = None
    reasoning_effort: str | None = None

    # Legacy defaults (backward compat)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    retry_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def matches(self, node: PipelineNode) -> bool:
        """Return ``True`` if this rule applies to *node*."""
        # CSS-style selectors
        if self.selector_type == "id":
            if node.name != self.selector_value:
                return False
        elif self.selector_type == "class":
            if self.selector_value not in node.classes:
                return False
        # selector_type == "universal" matches everything

        # Legacy selectors (backward compat)
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
        for attr in (
            "llm_model",
            "llm_provider",
            "reasoning_effort",
            "model",
            "temperature",
            "max_tokens",
            "timeout",
            "retry_count",
        ):
            value = getattr(self, attr)
            if value is not None:
                result[attr] = value
        result.update(self.extra)
        return result


@dataclass
class ModelStylesheet:
    """Ordered collection of :class:`StyleRule` entries.

    Rules are evaluated by specificity; higher specificity overrides
    lower.  Node-specific attributes always win.
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
                    llm_model=entry.get("llm_model"),
                    llm_provider=entry.get("llm_provider"),
                    reasoning_effort=entry.get("reasoning_effort"),
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
                            "llm_model",
                            "llm_provider",
                            "reasoning_effort",
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


def parse_stylesheet(css: str) -> ModelStylesheet:
    """Parse a CSS-like stylesheet string into a ModelStylesheet.

    Supported selectors:
        - ``*`` — universal (matches all nodes)
        - ``#name`` — ID selector (matches node by name)
        - ``.class`` — class selector (matches nodes with that class)

    Args:
        css: CSS-like stylesheet text.

    Returns:
        A :class:`ModelStylesheet` populated with parsed rules.
    """
    rules: list[StyleRule] = []
    rule_pattern = re.compile(r"([*#.][^\{]*?)\s*\{([^}]*)\}", re.DOTALL)

    for match in rule_pattern.finditer(css):
        selector_str = match.group(1).strip()
        body = match.group(2).strip()

        # Parse selector
        if selector_str == "*":
            selector_type = "universal"
            selector_value = "*"
            specificity = 0
        elif selector_str.startswith("#"):
            selector_type = "id"
            selector_value = selector_str[1:]
            specificity = 2
        elif selector_str.startswith("."):
            selector_type = "class"
            selector_value = selector_str[1:]
            specificity = 1
        else:
            continue  # Skip unknown selectors

        # Parse declarations
        props: dict[str, str] = {}
        for decl in body.split(";"):
            decl = decl.strip()
            if ":" in decl:
                prop, val = decl.split(":", 1)
                props[prop.strip()] = val.strip().strip('"').strip("'")

        known_props = {"llm_model", "llm_provider", "reasoning_effort"}
        rules.append(
            StyleRule(
                selector_type=selector_type,
                selector_value=selector_value,
                specificity=specificity,
                llm_model=props.get("llm_model"),
                llm_provider=props.get("llm_provider"),
                reasoning_effort=props.get("reasoning_effort"),
                extra={
                    k: v for k, v in props.items() if k not in known_props
                },
            )
        )

    return ModelStylesheet(rules=rules)


def apply_stylesheet(
    stylesheet: ModelStylesheet,
    node: PipelineNode,
    pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    """Resolve final attributes for *node* by layering stylesheet defaults.

    Evaluation order:
    1. Graph-level defaults from ``pipeline.metadata`` (lowest priority).
    2. Stylesheet rules sorted by specificity (lower first, so higher
       specificity overrides).
    3. Node's own attributes override everything.

    When the resolved attributes include ``llm_model``, ``llm_provider``,
    or ``reasoning_effort``, the corresponding :class:`PipelineNode` field
    is set directly — but only if the node doesn't already have it set.

    Args:
        stylesheet: The stylesheet to apply.
        node: The node to resolve attributes for.
        pipeline: Optional pipeline for graph-level default attributes.

    Returns:
        Merged attribute dict ready for handler consumption.
    """
    resolved: dict[str, Any] = {}
    # Graph-level defaults (lowest priority)
    if pipeline is not None:
        resolved.update(pipeline.metadata)
    # Sort by specificity (lower first, so higher specificity overrides)
    sorted_rules = sorted(stylesheet.rules, key=lambda r: r.specificity)
    for rule in sorted_rules:
        if rule.matches(node):
            resolved.update(rule.defaults())
    # Node-specific attributes always win
    resolved.update(node.attributes)
    # Set PipelineNode fields directly from resolved values
    if node.llm_model is None and "llm_model" in resolved:
        node.llm_model = resolved["llm_model"]
    if node.llm_provider is None and "llm_provider" in resolved:
        node.llm_provider = resolved["llm_provider"]
    if "reasoning_effort" in resolved and "reasoning_effort" not in node.attributes:
        node.reasoning_effort = resolved["reasoning_effort"]
    return resolved
