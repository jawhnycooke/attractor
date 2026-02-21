"""Condition evaluator for pipeline edge expressions.

Safely evaluates simple boolean expressions against a PipelineContext
using Python's ``ast`` module.  Supports comparisons, boolean logic,
and literal values without exposing ``eval`` or ``exec``.

Supported syntax
~~~~~~~~~~~~~~~~
- Comparisons: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
- Boolean ops: ``and``, ``or``, ``not``
- Identifiers: bare names resolve to ``context.get(name)``
- Literals: strings (``"quoted"``), ints, floats, booleans (``true``/``false``)

Examples::

    evaluate_condition('approved == true', ctx)
    evaluate_condition('exit_code != 0 and retries < 3', ctx)
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from attractor.pipeline.models import PipelineContext

# Operator mapping for safe comparison dispatch
_CMP_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
}

_BOOL_LITERALS = {"true": True, "false": False, "null": None, "none": None}


class ConditionError(Exception):
    """Raised when a condition expression cannot be parsed or evaluated."""


def evaluate_condition(expression: str, context: PipelineContext) -> bool:
    """Evaluate *expression* against *context* and return a boolean.

    Args:
        expression: The condition string (e.g. ``"exit_code == 0"``).
        context: Pipeline context providing variable values.

    Returns:
        The boolean result of the expression.

    Raises:
        ConditionError: If the expression is syntactically invalid or
            contains unsupported constructs.
    """
    if not expression or not expression.strip():
        return True  # empty condition is always true

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"Invalid condition syntax: {expression!r}") from exc

    return bool(_eval_node(tree.body, context))


def validate_condition_syntax(expression: str) -> str | None:
    """Check whether *expression* is syntactically valid.

    Returns:
        ``None`` if valid, or an error message string.
    """
    if not expression or not expression.strip():
        return None
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        _check_node(tree.body)
    except (SyntaxError, ConditionError) as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Internal AST walker
# ---------------------------------------------------------------------------


def _eval_node(node: ast.expr, ctx: PipelineContext) -> Any:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Compare):
        return _eval_compare(node, ctx)
    if isinstance(node, ast.BoolOp):
        return _eval_boolop(node, ctx)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        name = node.id
        if name in _BOOL_LITERALS:
            return _BOOL_LITERALS[name]
        return ctx.get(name)
    if isinstance(node, ast.Attribute):
        # support dotted names like "result.status"
        return _eval_attribute(node, ctx)
    raise ConditionError(f"Unsupported expression node: {type(node).__name__}")


def _eval_compare(node: ast.Compare, ctx: PipelineContext) -> bool:
    left = _eval_node(node.left, ctx)
    for op, comparator in zip(node.ops, node.comparators):
        right = _eval_node(comparator, ctx)
        op_func = _CMP_OPS.get(type(op))
        if op_func is None:
            raise ConditionError(f"Unsupported comparison: {type(op).__name__}")
        if not op_func(left, right):
            return False
        left = right
    return True


def _eval_boolop(node: ast.BoolOp, ctx: PipelineContext) -> bool:
    if isinstance(node.op, ast.And):
        return all(_eval_node(v, ctx) for v in node.values)
    if isinstance(node.op, ast.Or):
        return any(_eval_node(v, ctx) for v in node.values)
    raise ConditionError(f"Unsupported boolean op: {type(node.op).__name__}")


def _eval_attribute(node: ast.Attribute, ctx: PipelineContext) -> Any:
    """Resolve dotted attribute access (e.g. ``result.status``)."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        raise ConditionError("Unsupported attribute base")
    parts.reverse()
    key = ".".join(parts)
    return ctx.get(key)


def _check_node(node: ast.expr) -> None:
    """Validate that only supported AST nodes are present."""
    if isinstance(node, ast.Compare):
        _check_node(node.left)
        for comp in node.comparators:
            _check_node(comp)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_node(v)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _check_node(node.operand)
    elif isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        pass
    else:
        raise ConditionError(f"Unsupported expression node: {type(node).__name__}")
