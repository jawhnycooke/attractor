"""Condition evaluator for pipeline edge expressions.

Evaluates minimal boolean expressions per the attractor spec §10.
Uses a custom tokenizer/parser — no ``eval``, ``exec``, or ``ast.parse``.

Supported syntax (spec §10.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Operators: ``=`` (equality), ``!=`` (inequality)
- Conjunction: ``&&`` (AND, all clauses must be true)
- Keys: ``outcome``, ``preferred_label``, ``context.`` prefixed, or bare names
- Literals: strings (``"quoted"``), integers, booleans (``true``/``false``)

Examples::

    evaluate_condition('outcome=success', ctx)
    evaluate_condition('outcome=success && context.tests_passed=true', ctx)
    evaluate_condition('context.exit_code != 0', ctx)
"""

from __future__ import annotations

import enum
from typing import Any

from attractor.pipeline.models import PipelineContext

class ConditionError(Exception):
    """Raised when a condition expression cannot be parsed or evaluated."""


def evaluate_condition(
    expression: str,
    context: PipelineContext,
    extra_vars: dict[str, Any] | None = None,
) -> bool:
    """Evaluate *expression* against *context* and return a boolean.

    Implements the spec §10.5 evaluation algorithm: split on ``&&``,
    evaluate each clause independently, return ``True`` only if all
    clauses pass.

    Args:
        expression: The condition string (e.g. ``"outcome=success"``).
        context: Pipeline context providing variable values.
        extra_vars: Optional mapping of first-class variables such as
            ``outcome`` and ``preferred_label`` that take precedence
            over context lookups.

    Returns:
        The boolean result of the expression.

    Raises:
        ConditionError: If the expression is syntactically invalid or
            contains unsupported constructs.
    """
    if not expression or not expression.strip():
        return True  # empty condition is always true

    extras = extra_vars or {}
    clauses = expression.split("&&")

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if not _evaluate_clause(clause, context, extras):
            return False

    return True


def validate_condition_syntax(expression: str) -> str | None:
    """Check whether *expression* is syntactically valid.

    Returns:
        ``None`` if valid, or an error message string.
    """
    if not expression or not expression.strip():
        return None
    try:
        clauses = expression.split("&&")
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            _parse_clause(clause)
    except ConditionError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Internal evaluator
# ---------------------------------------------------------------------------


def _resolve_key(
    key: str,
    context: PipelineContext,
    extras: dict[str, Any],
) -> str:
    """Resolve a key to its string value per spec §10.4.

    Resolution order:
    1. ``outcome`` and ``preferred_label`` resolve from *extras* first.
    2. ``context.``-prefixed keys look up from context (with and without prefix).
    3. Bare names look up from *extras* first, then context.
    Missing keys return ``""``.
    """
    # First-class variables from extras
    if key == "outcome":
        val = extras.get("outcome")
        if val is not None:
            return _to_string(val)
        # Fall through to context lookup
    if key == "preferred_label":
        val = extras.get("preferred_label")
        if val is not None:
            return _to_string(val)
        # Fall through to context lookup

    # context.-prefixed keys
    if key.startswith("context."):
        suffix = key[len("context."):]
        # Try with full key first
        val = context.get(key)
        if val is not None:
            return _to_string(val)
        # Try without prefix
        val = context.get(suffix)
        if val is not None:
            return _to_string(val)
        return ""

    # Bare name: check extras, then context
    val = extras.get(key)
    if val is not None:
        return _to_string(val)
    val = context.get(key)
    if val is not None:
        return _to_string(val)
    return ""


def _to_string(value: Any) -> str:
    """Convert a value to its string representation for comparison."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _parse_literal(raw: str) -> str:
    """Parse a literal value string and return its normalized form.

    Handles quoted strings (strips quotes), boolean literals, and
    integers/bare values (returned as-is).
    """
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1]
    return raw


def _parse_clause(clause: str) -> tuple[str, str, str] | tuple[str,]:
    """Parse a single clause into (key, operator, literal) or (bare_key,).

    Raises ConditionError for unsupported operators or invalid syntax.
    """
    # Check for unsupported Python-style operators that users might try
    _reject_unsupported_operators(clause)

    # Check for != first (before =) to avoid partial matching
    if "!=" in clause:
        parts = clause.split("!=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        if not key:
            raise ConditionError(f"Invalid condition syntax: {clause!r}")
        return (key, "!=", value)

    if "=" in clause:
        parts = clause.split("=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        if not key:
            raise ConditionError(f"Invalid condition syntax: {clause!r}")
        return (key, "=", value)

    # Bare key — treated as truthy check
    key = clause.strip()
    if not key:
        raise ConditionError(f"Invalid condition syntax: {clause!r}")
    return (key,)


def _reject_unsupported_operators(clause: str) -> None:
    """Reject Python-style and unsupported operators."""
    # Check for Python-style == (spec uses single =)
    if "==" in clause:
        raise ConditionError(
            f"Unsupported operator '==' in condition: {clause!r}; "
            "use '=' for equality"
        )

    # Check for unsupported comparison operators
    # Be careful not to match != which is valid
    stripped = clause.replace("!=", "  ")  # mask valid !=
    for op in ("<=", ">="):
        if op in stripped:
            raise ConditionError(
                f"Unsupported operator '{op}' in condition: {clause!r}; "
                "only '=' and '!=' are allowed"
            )
    # Check < and > but not inside quoted strings
    _check_angle_brackets(stripped, clause)

    # Check for Python-style boolean operators
    # Only reject 'and'/'or'/'not' as standalone words
    tokens = clause.split()
    for token in tokens:
        if token in ("and", "or", "not"):
            raise ConditionError(
                f"Unsupported operator '{token}' in condition: {clause!r}; "
                "use '&&' for AND"
            )


def _check_angle_brackets(masked: str, original: str) -> None:
    """Check for bare < or > operators outside of quoted strings."""
    in_quotes = False
    quote_char = ""
    for ch in masked:
        if ch in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = ch
        elif ch == quote_char and in_quotes:
            in_quotes = False
        elif not in_quotes and ch in ("<", ">"):
            raise ConditionError(
                f"Unsupported operator '{ch}' in condition: {original!r}; "
                "only '=' and '!=' are allowed"
            )


def _evaluate_clause(
    clause: str,
    context: PipelineContext,
    extras: dict[str, Any],
) -> bool:
    """Evaluate a single clause per spec §10.5."""
    parsed = _parse_clause(clause)

    if len(parsed) == 1:
        # Bare key: truthy check
        resolved = _resolve_key(parsed[0], context, extras)
        return bool(resolved)

    key, op, raw_value = parsed  # type: ignore[misc]
    resolved = _resolve_key(key, context, extras)
    literal = _parse_literal(raw_value)

    if op == "=":
        return resolved == literal
    if op == "!=":
        return resolved != literal

    raise ConditionError(f"Unsupported operator: {op!r}")
