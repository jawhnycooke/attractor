# CLAUDE.md — Pipeline Subsystem

The pipeline subsystem implements a DAG-based execution engine. Workflows are defined as GraphViz DOT files, parsed into a `Pipeline` graph, validated, and executed node-by-node with condition-based routing and checkpoint-based resumability.

## Module Map

| Module | Purpose |
|---|---|
| `models.py` | Core data structures: `Pipeline`, `PipelineNode`, `PipelineEdge`, `NodeResult`, `PipelineContext`, `Checkpoint`. |
| `engine.py` | `PipelineEngine` — walks the DAG, dispatches handlers, evaluates edges, checkpoints after each node. |
| `parser.py` | `parse_dot_file()` — reads DOT files via `pydot`, extracts nodes/edges/attributes. |
| `handlers.py` | `NodeHandler` protocol, `HandlerRegistry`, built-in `CodergenHandler`. |
| `conditions.py` | Custom tokenizer/parser for edge condition expressions. Supports `=`, `!=`, `&&` only. No `eval`/`exec`. |
| `events.py` | Pipeline event system: `PipelineEventType` enum and `PipelineEvent` dataclass for execution observability. |
| `validator.py` | Static validation: start node, terminal nodes, handler types, edge references, reachability, cycles. |
| `state.py` | Checkpoint persistence: `save_checkpoint()`, `latest_checkpoint()`, `list_checkpoints()`. |
| `goals.py` | `GoalGate` — enforces required node completions and context conditions before terminal exit. |
| `stylesheet.py` | `ModelStylesheet` — rule-based default attributes (model, temperature, etc.) applied per-node. |
| `interviewer.py` | `Interviewer` protocol for human-in-the-loop. `CLIInterviewer` (Rich-based), `QueueInterviewer` (tests). |

## Key Patterns

- **Execution flow**: Start node → handler dispatch → context merge → edge evaluation → checkpoint → next node → repeat.
- **Edge routing**: Outgoing edges sorted by weight (higher = higher priority). First matching condition wins. Unconditional edges serve as fallback. No match on a non-terminal node is an error.
- **Condition expressions**: Evaluated via a custom tokenizer/parser — supports `=`, `!=`, and `&&` only. Explicitly rejects `==`, `<`, `>`, `and`, `or`, `not`. Variables resolve from `PipelineContext`. No arbitrary code execution.
- **Handler override routing**: A handler can set `NodeResult.next_node` to bypass edge evaluation and force a specific next node.
- **Blackboard pattern**: `PipelineContext` is a shared key-value store. Nodes read/write context values. Internal keys prefixed with `_` (e.g., `_last_error`, `_failed_node`).
- **Terminal detection**: Nodes with shape=Msquare (handler_type "exit"), name matching "exit"/"end" (case-insensitive), or no outgoing edges.
- **Start node detection**: Node with shape=Mdiamond (handler_type "start"), then case-insensitive name match for "start".
- **Checkpoint naming**: `checkpoint_{timestamp_ms}.json` in the configured checkpoint directory.
- **Stylesheet cascading**: Rules match by CSS-style selectors: `*` (universal), `#id` (node ID), `.class` (class name). Specificity order: universal < class < ID. Node-specific attributes always override stylesheet defaults.

## DOT File Format

```dot
digraph my_pipeline {
    start [shape=Mdiamond]
    implement [type="codergen" prompt="Fix the login bug"]
    review [type="codergen" prompt="Review the changes"]
    done [shape=Msquare]

    start -> implement
    implement -> review
    review -> done [condition="review_passed = true"]
    review -> implement [condition="review_passed = false" weight=2]
}
```

Node attributes: `type` (handler type override), `prompt`, `model`, `temperature`, `max_tokens`, `shape`, `max_retries`, `allow_partial`, `fidelity`, `timeout`.

**Note**: The DOT attribute is `type` (not `handler` or `handler_type`). The parser maps it to the internal `PipelineNode.handler_type` field. If `type` is absent, handler type is inferred from the node's `shape` attribute.
Edge attributes: `condition`, `weight` (higher = higher priority), `label`, `fidelity`, `thread_id`, `loop_restart`.

## Adding a New Handler

1. Implement the `NodeHandler` protocol: `async execute(node, context) -> NodeResult`.
2. Register it in `HandlerRegistry` with a handler type string.
3. Add the handler type to `validator.py`'s known types set.
4. Use it in DOT files via `type="your_type"`.
5. If it needs pipeline awareness, accept `pipeline` in the constructor (see `CodergenHandler`).

## Context Convention

- `_`-prefixed keys are engine-internal: `_last_error`, `_failed_node`, `_completed_nodes`, `_goal_gate_unmet`.
- User-defined keys are plain strings set by handlers via `NodeResult.context_updates`.
- Conditions reference context keys by bare name (no `$` or `{}`).
- Prompt interpolation uses `{key}` syntax within handler prompt strings.
