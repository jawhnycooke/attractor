# Status File Contract — Spec Compliance Gap

## Status: Not Implemented

This document describes the `status.json` file contract required by the
attractor specification. No part of this is currently implemented.

---

## Overview

Each pipeline node writes a `status.json` file to its stage directory after
handler execution completes. This file is the formal interface between handlers
and the engine's routing logic — it declares the outcome, routing hints, context
updates, and human-readable notes.

---

## File Location

```
{logs_root}/{node_id}/status.json
```

Where:
- `{logs_root}` is the run working directory (see [run-directory.md](run-directory.md))
- `{node_id}` is the pipeline node name (e.g., `"code_review"`, `"deploy"`)

The directory `{logs_root}/{node_id}/` must be created by the engine before
invoking the handler.

---

## Schema

```json
{
    "outcome": "success",
    "preferred_next_label": "approved",
    "suggested_next_ids": ["deploy", "rollback"],
    "context_updates": {
        "review.passed": true,
        "review.score": 8.5
    },
    "notes": "Code review passed with minor suggestions"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `outcome` | `string` | **Yes** | One of: `"success"`, `"partial_success"`, `"retry"`, `"fail"`, `"skipped"` |
| `preferred_next_label` | `string \| null` | No | Edge label to prioritize for the next transition (Step 2 of edge selection) |
| `suggested_next_ids` | `string[]` | No | Fallback target node IDs if label match fails (Step 3 of edge selection) |
| `context_updates` | `object` | No | Key-value pairs merged into the pipeline context after execution |
| `notes` | `string \| null` | No | Human-readable execution summary for logging and audit |

### Outcome Values

| Value | Meaning | Engine Behavior |
|-------|---------|-----------------|
| `"success"` | Handler completed successfully | Normal edge selection proceeds |
| `"partial_success"` | Completed with caveats | Treated as success for routing; `allow_partial` controls goal gates |
| `"retry"` | Transient failure, should retry | Engine decrements retry counter, re-executes if retries remain |
| `"fail"` | Permanent failure | Triggers failure routing: outcome=fail edge → retry_target → fallback |
| `"skipped"` | Handler chose not to execute | Normal edge selection; node not counted for goal gates |

---

## Write Timing

### Normal Flow

1. Engine creates `{logs_root}/{node_id}/` directory
2. Engine invokes handler with `logs_root` path
3. Handler executes and writes `status.json` to its stage directory
4. Engine reads `status.json` and processes the result
5. Engine merges `context_updates` into the pipeline context
6. Engine evaluates edges using `outcome`, `preferred_next_label`, `suggested_next_ids`

### Handler Crash / Exception

If the handler raises an exception before writing `status.json`, the engine
synthesizes a failure status file:

```json
{
    "outcome": "fail",
    "notes": "Handler exception: <exception message>"
}
```

### Auto-Status

When a node has `auto_status=true` and the handler does **not** write a
`status.json` file, the engine synthesizes:

```json
{
    "outcome": "success",
    "notes": "auto-status: handler completed without writing status"
}
```

This is useful for simple pass-through nodes that don't need explicit status
reporting.

---

## Implementation Requirements

### Engine Changes (`engine.py`)

1. Before handler dispatch, create the node's stage directory:
   ```python
   stage_dir = Path(logs_root) / node.name
   stage_dir.mkdir(parents=True, exist_ok=True)
   ```

2. Pass `logs_root` to the handler (via context or as a parameter).

3. After handler execution, check for `status.json`:
   ```python
   status_path = stage_dir / "status.json"
   if status_path.exists():
       status = json.loads(status_path.read_text())
       # Use status fields for routing
   elif node.auto_status:
       # Synthesize success
   else:
       # Use NodeResult from handler return value (current behavior)
   ```

4. On handler exception, write synthesized failure status.

### Handler Changes

Handlers that want to use the status file contract should write to
`{logs_root}/{node.name}/status.json`. This is **optional** — handlers can
continue to return `NodeResult` directly. The engine should support both
mechanisms:

- If `status.json` exists, it takes precedence
- Otherwise, the `NodeResult` return value is used

This allows gradual migration without breaking existing handlers.

### NodeResult ↔ status.json Mapping

The `NodeResult` dataclass already contains all the fields needed:

| NodeResult Field | status.json Field |
|-----------------|-------------------|
| `status` (OutcomeStatus) | `outcome` |
| `preferred_label` | `preferred_next_label` |
| `suggested_next_ids` | `suggested_next_ids` |
| `context_updates` | `context_updates` |
| `notes` | `notes` |

A utility function should handle bidirectional conversion:

```python
def node_result_to_status(result: NodeResult) -> dict[str, Any]:
    return {
        "outcome": result.status.value,
        "preferred_next_label": result.preferred_label,
        "suggested_next_ids": result.suggested_next_ids,
        "context_updates": result.context_updates,
        "notes": result.notes,
    }

def status_to_node_result(status: dict[str, Any]) -> NodeResult:
    return NodeResult(
        status=OutcomeStatus(status["outcome"]),
        preferred_label=status.get("preferred_next_label"),
        suggested_next_ids=status.get("suggested_next_ids", []),
        context_updates=status.get("context_updates", {}),
        notes=status.get("notes"),
    )
```

---

## Testing Requirements

1. **Engine writes synthesized status on handler crash** — verify file content
2. **Engine writes auto-status when handler doesn't** — verify `auto_status=true` behavior
3. **Engine reads handler-written status.json** — verify routing uses file content
4. **NodeResult ↔ status.json round-trip** — verify conversion fidelity
5. **status.json takes precedence over NodeResult** — verify when both exist
6. **Stage directory creation** — verify directory exists before handler runs
