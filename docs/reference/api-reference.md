# Attractor API Reference

**Version**: 0.1.0
**Python**: >= 3.11
**License**: Apache-2.0

Attractor is a non-interactive coding agent for software factories. It orchestrates LLM-driven workflows defined as GraphViz DOT pipelines, executing them through an autonomous agent with tool-use capabilities.

**Architecture layers**:

```
CLI (click)
 └─ pipeline/ ── Parses DOT → validates → executes node graph
      └─ agent/ ── CodergenHandler invokes Session for LLM-driven coding
           └─ llm/ ── Routes requests to provider adapters (Anthropic, OpenAI, Gemini)
```

---

## Table of Contents

1. [CLI Reference](#1-cli-reference)
2. [Pipeline API Reference](#2-pipeline-api-reference)
3. [Agent API Reference](#3-agent-api-reference)
4. [LLM Client API Reference](#4-llm-client-api-reference)

---

## 1. CLI Reference

**Entry point**: `attractor` (installed via `pyproject.toml` `[project.scripts]`)

**Synopsis**:

```bash
attractor [--version] [--help] <subcommand> [OPTIONS] [ARGUMENTS]
```

### Global Options

#### --version

Show the installed version and exit.

#### --help

Show help text and exit.

---

### `attractor run`

Execute a pipeline from a DOT file.

**Synopsis**:

```bash
attractor run [OPTIONS] PIPELINE_DOT
```

**Arguments**:

- `PIPELINE_DOT` (required): Path to a GraphViz DOT file. Must exist on the filesystem.

**Options**:

#### --model MODEL

- **Type**: string
- **Required**: No
- **Default**: None
- **Description**: Default model identifier for `codergen` nodes (e.g. `gpt-4o`, `claude-opus-4-6`). Applied as a stylesheet rule matching all `codergen` handler_type nodes. Node-level `model` attributes override this value.

#### --provider PROVIDER

- **Type**: string
- **Required**: No
- **Default**: None
- **Description**: LLM provider name. Currently accepted for future use; model routing is automatic based on the model string prefix.

#### --verbose, -v

- **Type**: flag
- **Default**: False
- **Description**: Enable debug-level logging via `RichHandler`. Without this flag, logging is at INFO level.

#### --checkpoint-dir PATH

- **Type**: directory path (string)
- **Default**: `.attractor/checkpoints`
- **Description**: Directory where checkpoint files are written after each completed node. Created automatically if it does not exist. Files are named `checkpoint_{timestamp_ms}.json`.

**Behavior**:

1. Parses `PIPELINE_DOT` via `parse_dot_file()`.
2. Validates the pipeline via `validate_pipeline()`. Exits with code 1 if any ERROR-level findings exist. Prints WARNING-level findings.
3. Builds a `ModelStylesheet` from CLI options.
4. Constructs a `HandlerRegistry` with all built-in handlers.
5. Runs `PipelineEngine.run(pipeline)` asynchronously.
6. Prints the final `PipelineContext` as a Rich table (keys starting with `_` are hidden).

**Exit codes**:

| Code | Condition |
|------|-----------|
| 0 | Pipeline completed successfully |
| 1 | Parse error, validation error, or unrecoverable engine error |

**Example**:

```bash
attractor run pipeline.dot --model gpt-4o --verbose --checkpoint-dir /tmp/checkpoints
```

---

### `attractor validate`

Validate a pipeline DOT file without executing it.

**Synopsis**:

```bash
attractor validate [OPTIONS] PIPELINE_DOT
```

**Arguments**:

- `PIPELINE_DOT` (required): Path to a GraphViz DOT file. Must exist.

**Options**:

#### --strict

- **Type**: flag
- **Default**: False
- **Description**: Treat WARNING-level findings as errors. With `--strict`, any finding (error or warning) causes exit code 1.

**Behavior**:

1. Parses the DOT file.
2. Runs all validation checks.
3. Prints a Rich table with columns: Level, Location, Message.
4. Exits 0 if valid (or warnings-only without `--strict`).

**Exit codes**:

| Code | Condition |
|------|-----------|
| 0 | No errors (warnings permitted without `--strict`) |
| 1 | Parse error, ERROR findings, or WARNING findings with `--strict` |

**Example**:

```bash
attractor validate pipeline.dot --strict
```

---

### `attractor resume`

Resume a pipeline from a checkpoint file.

**Synopsis**:

```bash
attractor resume [OPTIONS] CHECKPOINT_PATH
```

**Arguments**:

- `CHECKPOINT_PATH` (required): Path to a checkpoint JSON file. Must exist.

**Options**:

#### --verbose, -v

- **Type**: flag
- **Default**: False
- **Description**: Enable debug-level logging.

#### --pipeline-dot PATH

- **Type**: file path
- **Required**: Yes (at runtime — the flag is optional in declaration but the command exits with code 1 if omitted)
- **Default**: None
- **Description**: Path to the DOT file for the pipeline being resumed. The checkpoint stores metadata about the original pipeline but does not embed the full DOT source.

**Behavior**:

1. Loads the checkpoint from `CHECKPOINT_PATH` via `Checkpoint.load_from_file()`.
2. Parses the pipeline from `--pipeline-dot`.
3. Runs `PipelineEngine.run(pipeline, checkpoint=cp)`, which resumes from `checkpoint.current_node`.

**Exit codes**:

| Code | Condition |
|------|-----------|
| 0 | Pipeline completed after resuming |
| 1 | Checkpoint load error, parse error, or missing `--pipeline-dot` |

**Example**:

```bash
attractor resume .attractor/checkpoints/checkpoint_1700000000000.json \
    --pipeline-dot pipeline.dot --verbose
```

---

### Environment Variables

| Variable | Provider | Required for |
|----------|----------|--------------|
| `ANTHROPIC_API_KEY` | Anthropic | `claude-*` models |
| `OPENAI_API_KEY` | OpenAI | `gpt-*`, `o1`, `o3`, `o4` models |
| `GOOGLE_API_KEY` | Google | `gemini-*` models |

Variables are loaded from the process environment. A `.env` file is not automatically read — callers must source it before invoking `attractor` or use a tool such as `python-dotenv`.

---

## 2. Pipeline API Reference

**Module prefix**: `attractor.pipeline`

---

### `Pipeline`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class Pipeline:
    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[PipelineEdge] = field(default_factory=list)
    start_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

Complete pipeline definition parsed from a DOT file.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Pipeline identifier — typically the DOT graph name, or the file stem if unnamed. |
| `nodes` | `dict[str, PipelineNode]` | Mapping of node name to `PipelineNode`. |
| `edges` | `list[PipelineEdge]` | All directed edges in the graph. |
| `start_node` | `str` | Name of the designated start node. |
| `metadata` | `dict[str, Any]` | Graph-level attributes from the DOT file. |

**Methods**:

#### `outgoing_edges(node_name: str) -> list[PipelineEdge]`

Returns edges originating from `node_name`, sorted ascending by `edge.priority`.

#### `incoming_edges(node_name: str) -> list[PipelineEdge]`

Returns edges whose `target` is `node_name`. Not sorted.

---

### `PipelineNode`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class PipelineNode:
    name: str
    handler_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    is_start: bool = False
    is_terminal: bool = False
```

A single node in the pipeline graph.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Unique identifier for the node. Matches the DOT node name. |
| `handler_type` | `str` | — | Dispatch key for the handler registry (e.g. `"codergen"`, `"human_gate"`, `"tool"`). Defaults to `"codergen"` if the `handler` attribute is absent from the DOT node. |
| `attributes` | `dict[str, Any]` | `{}` | Handler-specific configuration extracted from DOT attributes and/or stylesheet overrides. Common keys: `prompt`, `model`, `temperature`, `max_tokens`, `timeout`, `command`. |
| `is_start` | `bool` | `False` | `True` for the pipeline entry point. |
| `is_terminal` | `bool` | `False` | `True` for nodes that end pipeline execution. |

**DOT attribute mapping**:

| DOT attribute | Field / effect |
|---------------|----------------|
| `handler` | Sets `handler_type`. Default: `"codergen"`. |
| `start=true` | Sets `is_start = True`. |
| `terminal=true` | Sets `is_terminal = True`. |
| All other attributes | Stored in `attributes`. |

---

### `PipelineEdge`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class PipelineEdge:
    source: str
    target: str
    condition: str | None = None
    label: str = ""
    priority: int = 0
```

A directed edge between two pipeline nodes.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `str` | — | Name of the originating node. |
| `target` | `str` | — | Name of the destination node. |
| `condition` | `str \| None` | `None` | Condition expression evaluated against `PipelineContext`. `None` means unconditional (default/fallback edge). |
| `label` | `str` | `""` | Human-readable label for visualization purposes. |
| `priority` | `int` | `0` | Ordering hint when multiple edges share a source. Lower values are evaluated first. |

---

### `PipelineContext`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class PipelineContext:
    data: dict[str, Any] = field(default_factory=dict)
```

Shared key-value state store (blackboard) for pipeline execution. All nodes read from and write to a single shared context.

**Internal keys** (set by the engine, prefixed with `_`):

| Key | Type | Set when |
|-----|------|----------|
| `_last_error` | `str` | A node's `result.success` is `False`. |
| `_failed_node` | `str` | A node's `result.success` is `False`. |
| `_completed_nodes` | `list[str]` | After pipeline finishes. |
| `_goal_gate_unmet` | `list[str]` | If a `GoalGate` check fails at terminal exit. |
| `_supervisor_iteration` | `int` | Set by `SupervisorHandler` each iteration. |
| `_supervisor_iterations` | `int` | Total iterations used by `SupervisorHandler`. |

**Methods**:

#### `get(key: str, default: Any = None) -> Any`

Return the value for `key`, or `default` if absent.

#### `set(key: str, value: Any) -> None`

Set `key` to `value`.

#### `has(key: str) -> bool`

Return `True` if `key` is present.

#### `delete(key: str) -> None`

Remove `key`. No-op if absent.

#### `update(updates: dict[str, Any]) -> None`

Merge `updates` into the context, overwriting existing keys.

#### `to_dict() -> dict[str, Any]`

Return a shallow copy of the internal data dict.

#### `from_dict(data: dict[str, Any]) -> PipelineContext` (classmethod)

Construct a `PipelineContext` from a plain dict.

#### `create_scope(prefix: str) -> PipelineContext`

Create an empty child context for branch isolation. The `prefix` argument is used only when merging back.

#### `merge_scope(scope: PipelineContext, prefix: str) -> None`

Merge a scoped context back into this context. Keys from `scope` are stored as `"{prefix}.{key}"`.

---

### `NodeResult`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class NodeResult:
    success: bool
    output: Any = None
    error: str | None = None
    next_node: str | None = None
    context_updates: dict[str, Any] = field(default_factory=dict)
```

Result returned by a node handler after execution.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `success` | `bool` | — | Whether the handler completed without error. |
| `output` | `Any` | `None` | Arbitrary output payload (stored in context by some handlers). |
| `error` | `str \| None` | `None` | Error description when `success` is `False`. |
| `next_node` | `str \| None` | `None` | Explicit routing override — bypasses edge evaluation. If set, the engine goes directly to this node. |
| `context_updates` | `dict[str, Any]` | `{}` | Key-value pairs merged into `PipelineContext` after execution. |

---

### `Checkpoint`

**Module**: `attractor.pipeline.models`

```python
@dataclass
class Checkpoint:
    pipeline_name: str
    current_node: str
    context: PipelineContext
    completed_nodes: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
```

Serializable execution snapshot for resume-on-failure.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `pipeline_name` | `str` | Name of the pipeline being executed. |
| `current_node` | `str` | Node that was about to execute (or just completed) when the checkpoint was written. |
| `context` | `PipelineContext` | Full pipeline context at checkpoint time. |
| `completed_nodes` | `list[str]` | Ordered list of nodes that finished successfully. |
| `timestamp` | `float` | UNIX epoch (seconds) when the checkpoint was created. |

**Serialization methods**:

#### `to_dict() -> dict[str, Any]`

Serialize to a JSON-serializable dict with keys: `pipeline_name`, `current_node`, `context`, `completed_nodes`, `timestamp`.

#### `from_dict(data: dict[str, Any]) -> Checkpoint` (classmethod)

Deserialize from a dict produced by `to_dict()`.

#### `save_to_file(path: str | Path) -> None`

Write JSON to `path`. Creates parent directories with `parents=True, exist_ok=True`.

#### `load_from_file(path: str | Path) -> Checkpoint` (classmethod)

Read and deserialize a checkpoint JSON file.

**File naming**: The engine writes files as `checkpoint_{timestamp_ms}.json` where `timestamp_ms = int(timestamp * 1000)`.

---

### `PipelineEngine`

**Module**: `attractor.pipeline.engine`

```python
class PipelineEngine:
    def __init__(
        self,
        registry: HandlerRegistry | None = None,
        stylesheet: ModelStylesheet | None = None,
        checkpoint_dir: str | Path | None = None,
        goal_gate: GoalGate | None = None,
        max_steps: int = 1000,
    ) -> None: ...
```

Single-threaded pipeline execution engine. Walks the DAG from the start node.

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | `HandlerRegistry \| None` | `None` | Handler registry. If `None`, `create_default_registry()` is called at run time. |
| `stylesheet` | `ModelStylesheet \| None` | `None` | Stylesheet for attribute defaults. Defaults to an empty `ModelStylesheet`. |
| `checkpoint_dir` | `str \| Path \| None` | `None` | Directory for checkpoint files. Checkpointing is disabled when `None`. |
| `goal_gate` | `GoalGate \| None` | `None` | Optional gate checked before allowing terminal exit. |
| `max_steps` | `int` | `1000` | Maximum node executions before the engine halts with a warning. |

**Methods**:

#### `run(pipeline, context=None, checkpoint=None) -> PipelineContext` (async)

```python
async def run(
    self,
    pipeline: Pipeline,
    context: PipelineContext | None = None,
    checkpoint: Checkpoint | None = None,
) -> PipelineContext
```

Execute `pipeline` and return the final context.

- `pipeline`: The pipeline definition to execute.
- `context`: Initial context. Defaults to an empty `PipelineContext`.
- `checkpoint`: If provided, resumes from `checkpoint.current_node` with `checkpoint.context`.

**Returns**: `PipelineContext` after pipeline completion.

**Raises**: `EngineError` on unrecoverable errors (missing handler, unknown next node).

**Execution loop per step**:

1. Apply stylesheet defaults to node attributes.
2. Dispatch to handler via `registry.get(node.handler_type)`.
3. Merge `result.context_updates` into context.
4. Append node name to completed list.
5. Write checkpoint (if `checkpoint_dir` is set).
6. Determine next node via `_resolve_next()`.

**Routing resolution order**:

1. `result.next_node` (explicit override) — used if set.
2. `node.is_terminal` — stops if True.
3. Outgoing edges sorted by `priority` (ascending) — first condition that evaluates True wins.
4. Unconditional edge — used as fallback if no conditions matched.
5. No outgoing edges — implicitly terminal, stops.

#### `run_sub_pipeline(pipeline_name, context) -> PipelineContext` (async)

Hook for `SupervisorHandler`. The base implementation logs a warning and returns the context unchanged. Override or subclass to provide sub-pipeline resolution.

---

### `EngineError`

**Module**: `attractor.pipeline.engine`

```python
class EngineError(Exception): ...
```

Raised for unrecoverable engine failures: missing handler type, unknown next node.

---

### `NodeHandler` protocol

**Module**: `attractor.pipeline.handlers`

```python
@runtime_checkable
class NodeHandler(Protocol):
    async def execute(
        self, node: PipelineNode, context: PipelineContext
    ) -> NodeResult: ...
```

Protocol that all node handlers must satisfy. Structural typing — no inheritance required.

---

### `HandlerRegistry`

**Module**: `attractor.pipeline.handlers`

```python
class HandlerRegistry:
    def __init__(self) -> None: ...
```

Maps handler-type strings to `NodeHandler` instances.

**Methods**:

#### `register(handler_type: str, handler: NodeHandler) -> None`

Register a handler for `handler_type`. Overwrites any existing registration.

#### `get(handler_type: str) -> NodeHandler | None`

Return the handler for `handler_type`, or `None` if not registered.

#### `has(handler_type: str) -> bool`

Return `True` if `handler_type` is registered.

#### `registered_types` (property) `-> list[str]`

Return all registered handler type strings.

---

### `create_default_registry()`

**Module**: `attractor.pipeline.handlers`

```python
def create_default_registry(
    pipeline: Pipeline | None = None,
    interviewer: Any = None,
) -> HandlerRegistry
```

Create a `HandlerRegistry` pre-loaded with all built-in handlers.

**Registered handler types**:

| Type string | Handler class |
|-------------|---------------|
| `"codergen"` | `CodergenHandler()` |
| `"human_gate"` | `HumanGateHandler(interviewer=interviewer)` |
| `"conditional"` | `ConditionalHandler(pipeline=pipeline)` |
| `"parallel"` | `ParallelHandler(registry=registry, pipeline=pipeline)` |
| `"tool"` | `ToolHandler()` |
| `"supervisor"` | `SupervisorHandler()` |

---

### `CodergenHandler`

**Module**: `attractor.pipeline.handlers`

Invokes the Attractor coding agent to execute a prompt. Falls back to an echo stub if `attractor.agent` cannot be imported.

**Node attributes consumed**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Prompt sent to the agent session. Supports `{key}` interpolation from context. |
| `model` | `str` | Model identifier. Passed to `Session(model=model)`. Empty string uses the session default. |

**Context updates set**:

| Key | Value |
|-----|-------|
| `last_codergen_output` | The output string returned by the agent session, or `"[stub] {prompt}"` in fallback mode. |

**Fallback behavior**: If `from attractor.agent import Session` raises `ImportError`, the handler returns `success=True` with a stub output string and logs a WARNING.

---

### `HumanGateHandler`

**Module**: `attractor.pipeline.handlers`

Presents a prompt to a human interviewer and gates on approval.

**Constructor**:

```python
HumanGateHandler(interviewer: Any = None)
```

**Node attributes consumed**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | `"Approve this step?"` | Question shown to the interviewer. |

**Context updates set**:

| Key | Value |
|-----|-------|
| `approved` | `bool` — `True` if approved, `False` if rejected, `True` if no interviewer (auto-approve). |

**Auto-approve behavior**: If no interviewer is configured (neither constructor arg nor `context.get("_interviewer")`), the handler logs a WARNING and approves automatically.

---

### `ConditionalHandler`

**Module**: `attractor.pipeline.handlers`

Evaluates outgoing edge conditions and sets `result.next_node` to the first matching target. Does not perform work itself.

**Constructor**:

```python
ConditionalHandler(pipeline: Pipeline | None = None)
```

---

### `ParallelHandler`

**Module**: `attractor.pipeline.handlers`

Fan-out execution across multiple sub-paths via `asyncio.gather`.

**Constructor**:

```python
ParallelHandler(
    registry: HandlerRegistry | None = None,
    pipeline: Pipeline | None = None,
)
```

**Node attributes consumed**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `branches` | `str \| list[str]` | Comma-separated string or list of node names to execute in parallel. |

**Context updates set**:

| Key | Value |
|-----|-------|
| `parallel_results` | `dict[branch_name, output]` for each branch. |
| `{branch_name}.{key}` | Each branch's `context_updates` are written with the branch name as prefix. |

---

### `ToolHandler`

**Module**: `attractor.pipeline.handlers`

Executes a shell command and captures exit code and output.

**Node attributes consumed**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | `str` | — | Shell command to execute. Required. Supports `{key}` context interpolation. |
| `timeout` | `float` | `300` | Command timeout in seconds. |

**Context updates set**:

| Key | Value |
|-----|-------|
| `exit_code` | `int` — process return code (`-1` on timeout). |
| `stdout` | `str` — standard output. |
| `stderr` | `str` — standard error. |

**Success condition**: `exit_code == 0`.

---

### `SupervisorHandler`

**Module**: `attractor.pipeline.handlers`

Iterative refinement loop. Executes a named sub-pipeline repeatedly until a condition is met or `max_iterations` is reached.

**Constructor**:

```python
SupervisorHandler(engine: Any = None)
```

**Node attributes consumed**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `sub_pipeline` | `str` | — | Name of the sub-pipeline to run. Required. |
| `max_iterations` | `int` | `5` | Maximum loop iterations. |
| `done_condition` | `str` | `""` | Condition expression evaluated against context after each iteration. Loop exits early when `True`. |

**Context writes**:

| Key | Value |
|-----|-------|
| `_supervisor_iteration` | `int` — current iteration number (1-based), set before each sub-pipeline run. |
| `_supervisor_iterations` | `int` — total iterations used, set on completion. |

---

### `parse_dot_file()`

**Module**: `attractor.pipeline.parser`

```python
def parse_dot_file(path: str | Path) -> Pipeline
```

Parse a DOT file at `path` and return a `Pipeline`.

**Parameters**:

- `path` (`str | Path`): Path to a `.dot` file.

**Returns**: `Pipeline`

**Raises**:

- `ParseError`: If the file cannot be read or the DOT content is invalid.
- `FileNotFoundError`: If `path` does not exist.

---

### `parse_dot_string()`

**Module**: `attractor.pipeline.parser`

```python
def parse_dot_string(dot_content: str, name: str = "pipeline") -> Pipeline
```

Parse a DOT string directly.

**Parameters**:

- `dot_content` (`str`): Raw DOT source text.
- `name` (`str`): Fallback pipeline name if the graph has no name. Default: `"pipeline"`.

**Returns**: `Pipeline`

**Raises**: `ParseError` if the DOT content is invalid or empty.

---

### `ParseError`

**Module**: `attractor.pipeline.parser`

```python
class ParseError(Exception): ...
```

Raised when a DOT file cannot be parsed into a valid pipeline.

---

### Start node detection

Precedence (evaluated in order):

1. Node with `start=true` attribute.
2. Node named `"start"`.
3. If neither exists: `ParseError` is raised.

### Terminal node detection

A node is terminal if:

- It has `terminal=true` attribute, **or**
- It has no outgoing edges (implicitly terminal — set by `_mark_terminals()`).

---

### `validate_pipeline()`

**Module**: `attractor.pipeline.validator`

```python
def validate_pipeline(pipeline: Pipeline) -> list[ValidationError]
```

Run all validation checks on `pipeline`. Returns a list of `ValidationError` findings, possibly empty.

**Checks performed** (in order):

| Check | Produces |
|-------|----------|
| Start node exists and is in `pipeline.nodes` | ERROR |
| At least one terminal node | ERROR |
| All `handler_type` values are in the known set | ERROR |
| Required attributes present per handler type | ERROR |
| All edge source/target node names exist | ERROR |
| All edge condition expressions are syntactically valid | ERROR |
| All nodes reachable from start node | WARNING |
| Cycles not involving a `supervisor` node | WARNING |

**Required attributes per handler type**:

| Handler type | Required attributes |
|-------------|---------------------|
| `"tool"` | `command` |

---

### `has_errors()`

**Module**: `attractor.pipeline.validator`

```python
def has_errors(findings: list[ValidationError]) -> bool
```

Return `True` if any finding has `level == ValidationLevel.ERROR`.

---

### `ValidationLevel`

**Module**: `attractor.pipeline.validator`

```python
class ValidationLevel(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
```

---

### `ValidationError`

**Module**: `attractor.pipeline.validator`

```python
@dataclass
class ValidationError:
    level: ValidationLevel
    message: str
    node_name: str | None = None
    edge: PipelineEdge | None = None
```

A single validation finding.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `level` | `ValidationLevel` | Severity: `ERROR` or `WARNING`. |
| `message` | `str` | Human-readable description. |
| `node_name` | `str \| None` | Name of the offending node, if applicable. |
| `edge` | `PipelineEdge \| None` | The offending edge, if applicable. |

**`__str__`**: Formats as `"[LEVEL] (location) message"`.

---

### Known handler types (validator)

The following `handler_type` strings are recognized by the built-in validator:

```
"codergen"
"human_gate"
"conditional"
"parallel"
"tool"
"supervisor"
```

Custom handler types registered at runtime will produce ERROR findings during static validation unless the validator is extended.

---

### `GoalGate`

**Module**: `attractor.pipeline.goals`

```python
@dataclass
class GoalGate:
    required_nodes: list[str] = field(default_factory=list)
    context_conditions: list[str] = field(default_factory=list)
```

Gate that must be satisfied before pipeline completion is allowed. Checked by `PipelineEngine` before exiting at a terminal node.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `required_nodes` | `list[str]` | Node names that must appear in the completed set. |
| `context_conditions` | `list[str]` | Condition expressions that must all evaluate to `True` against the final context. Uses the same syntax as edge conditions. |

**Methods**:

#### `check(completed_nodes: list[str], context: PipelineContext) -> bool`

Return `True` if all required nodes have completed and all context conditions hold.

#### `unmet_requirements(completed_nodes: list[str], context: PipelineContext) -> list[str]`

Return a list of human-readable strings describing each unmet requirement. Empty list means all requirements are met.

---

### `ModelStylesheet`

**Module**: `attractor.pipeline.stylesheet`

```python
@dataclass
class ModelStylesheet:
    rules: list[StyleRule] = field(default_factory=list)
```

Ordered collection of `StyleRule` entries. Rules are evaluated top-to-bottom; later rules override earlier ones for conflicting keys. Node-specific attributes always win over stylesheet defaults.

**Methods**:

#### `from_dict(data: dict[str, Any]) -> ModelStylesheet` (classmethod)

Build a stylesheet from a dict (e.g. parsed YAML/JSON).

**Expected format**:

```python
{
    "rules": [
        {
            "handler_type": "codergen",
            "model": "gpt-4o",
            "temperature": 0.2,
        },
        {
            "name_pattern": "test_*",
            "timeout": 120,
        },
    ]
}
```

---

### `StyleRule`

**Module**: `attractor.pipeline.stylesheet`

```python
@dataclass
class StyleRule:
    # Matching criteria
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
```

A single rule that applies default attribute values to matching nodes.

**Matching fields** (all specified criteria must match):

| Field | Type | Description |
|-------|------|-------------|
| `handler_type` | `str \| None` | Exact match on `node.handler_type`. |
| `name_pattern` | `str \| None` | `fnmatch` glob pattern matched against `node.name`. |
| `match_attributes` | `dict[str, Any]` | All key-value pairs must be present and equal in `node.attributes`. |

**Default fields applied to matching nodes**:

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str \| None` | Model identifier. |
| `temperature` | `float \| None` | Sampling temperature. |
| `max_tokens` | `int \| None` | Maximum output tokens. |
| `timeout` | `float \| None` | Timeout in seconds. |
| `retry_count` | `int \| None` | Retry count. |
| `extra` | `dict[str, Any]` | Additional arbitrary key-value defaults. |

**Methods**:

#### `matches(node: PipelineNode) -> bool`

Return `True` if this rule applies to `node`.

#### `defaults() -> dict[str, Any]`

Return only the non-`None` default values as a dict, merged with `extra`.

---

### `apply_stylesheet()`

**Module**: `attractor.pipeline.stylesheet`

```python
def apply_stylesheet(
    stylesheet: ModelStylesheet, node: PipelineNode
) -> dict[str, Any]
```

Resolve final attributes for `node` by layering stylesheet defaults.

**Returns**: Merged attribute dict: stylesheet defaults (later rules override earlier) overridden by the node's own attributes.

---

### `Interviewer` protocol

**Module**: `attractor.pipeline.interviewer`

```python
@runtime_checkable
class Interviewer(Protocol):
    async def ask(self, prompt: str, options: list[str] | None = None) -> str: ...
    async def confirm(self, prompt: str) -> bool: ...
    async def inform(self, message: str) -> None: ...
```

Protocol for human-in-the-loop interaction used by `HumanGateHandler`.

**Methods**:

| Method | Description |
|--------|-------------|
| `ask(prompt, options=None)` | Ask a free-form or multiple-choice question. If `options` is provided, returns one of the option strings. |
| `confirm(prompt)` | Ask for yes/no confirmation. Returns `bool`. |
| `inform(message)` | Display an informational message. No response expected. |

---

### `CLIInterviewer`

**Module**: `attractor.pipeline.interviewer`

```python
class CLIInterviewer:
    def __init__(self, console: Console | None = None) -> None: ...
```

Interactive interviewer using `rich` for terminal formatting. Blocks on `asyncio.to_thread` for stdin reads.

- `ask()` with `options`: Presents a numbered menu; returns the selected option string.
- `ask()` without `options`: Free-form `Prompt.ask`.
- `confirm()`: `Confirm.ask` returning `bool`.
- `inform()`: Prints a Rich `Panel` with `border_style="blue"`.

---

### `QueueInterviewer`

**Module**: `attractor.pipeline.interviewer`

```python
class QueueInterviewer:
    def __init__(self) -> None:
        self.responses: asyncio.Queue[str]
        self.messages: list[str]
```

Programmatic interviewer for tests and automation. Pre-load responses into `self.responses` before execution.

- `ask()`: Returns `await self.responses.get()`.
- `confirm()`: Returns `response.lower() in ("yes", "y", "true", "1")`.
- `inform()`: Appends `message` to `self.messages`.

---

### Checkpoint state functions

**Module**: `attractor.pipeline.state`

#### `save_checkpoint(checkpoint, directory) -> Path`

```python
def save_checkpoint(checkpoint: Checkpoint, directory: str | Path) -> Path
```

Persist `checkpoint` to `directory` with a timestamp-based filename. Creates the directory if needed. Returns the `Path` to the written file.

**File name format**: `checkpoint_{int(timestamp * 1000)}.json`

#### `list_checkpoints(directory) -> list[Path]`

```python
def list_checkpoints(directory: str | Path) -> list[Path]
```

Return all `checkpoint_*.json` files in `directory`, sorted newest first. Returns `[]` if directory does not exist.

#### `latest_checkpoint(directory) -> Checkpoint | None`

```python
def latest_checkpoint(directory: str | Path) -> Checkpoint | None
```

Load and return the most recent checkpoint from `directory`, or `None` if none exist.

---

### Condition expression syntax

**Module**: `attractor.pipeline.conditions`

Conditions are evaluated by `evaluate_condition(expression, context)` using Python's `ast` module. No `eval` or `exec` is used.

**Supported operators**:

| Operator | Type | Example |
|----------|------|---------|
| `==` | Comparison | `exit_code == 0` |
| `!=` | Comparison | `status != "failed"` |
| `<` | Comparison | `retries < 3` |
| `>` | Comparison | `score > 90` |
| `<=` | Comparison | `attempts <= 5` |
| `>=` | Comparison | `confidence >= 0.8` |
| `and` | Boolean | `approved == true and exit_code == 0` |
| `or` | Boolean | `status == "done" or skip == true` |
| `not` | Unary | `not failed` |

**Variable resolution**:

- Bare names (identifiers) resolve to `context.get(name)`.
- Dotted names (e.g. `result.status`) resolve to `context.get("result.status")`.
- `true`, `false` → `True`, `False`.
- `null`, `none` → `None`.
- String literals: `"double-quoted"`.
- Integer and float literals: `42`, `3.14`.

**Empty condition**: An empty or whitespace-only condition always evaluates to `True`.

**Unsupported**: Function calls, subscripts, lambda, comprehensions, arithmetic, bitwise operators. These raise `ConditionError`.

#### `evaluate_condition()`

```python
def evaluate_condition(expression: str, context: PipelineContext) -> bool
```

**Raises**: `ConditionError` if the expression is syntactically invalid or contains unsupported constructs.

#### `validate_condition_syntax()`

```python
def validate_condition_syntax(expression: str) -> str | None
```

Static syntax check. Returns `None` if valid, or an error message string.

#### `ConditionError`

```python
class ConditionError(Exception): ...
```

Raised when a condition expression cannot be parsed or evaluated.

---

## 3. Agent API Reference

**Module prefix**: `attractor.agent`

---

### `Session`

**Module**: `attractor.agent.session`

```python
class Session:
    def __init__(
        self,
        profile: ProviderProfile,
        environment: ExecutionEnvironment,
        config: SessionConfig,
        llm_client: LLMClientProtocol,
    ) -> None: ...
```

Top-level session managing conversation history and the agentic loop.

**Constructor parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile` | `ProviderProfile` | Provider-specific tool set and system prompt template. |
| `environment` | `ExecutionEnvironment` | Execution environment for file and shell operations. |
| `config` | `SessionConfig` | Session configuration. |
| `llm_client` | `LLMClientProtocol` | LLM client implementing `complete()`. |

**Properties**:

#### `state -> SessionState`

Current session state. Read-only.

#### `conversation_history -> list[Message]`

Copy of the current conversation history.

**Methods**:

#### `submit(user_input: str) -> AsyncIterator[AgentEvent]` (async generator)

Submit user input and yield events as the agent works.

- Transitions state: `IDLE → RUNNING → IDLE`.
- Emits `SESSION_START` at the beginning, `SESSION_END` at the end.
- Runs the agentic loop as a background `asyncio.Task` and yields events as they arrive.
- Raises `RuntimeError` if the session is `CLOSED`.

**Example**:

```python
async for event in session.submit("Fix the bug in auth.py"):
    print(event.type, event.data)
```

#### `follow_up(message: str) -> None`

Queue a message for processing after the current input completes. Messages are consumed in order after the primary submit loop finishes.

#### `steer(message: str) -> None`

Queue a steering message to inject after the current tool round, redirecting the agent mid-execution. (Reaches into the current loop via `AgentLoop.queue_steering()`.)

#### `set_reasoning_effort(effort: ReasoningEffort) -> None`

Change the reasoning effort for the next LLM call.

#### `shutdown() -> None` (async)

Graceful shutdown. Sets state to `CLOSED` and calls `environment.cleanup()`.

---

### `SessionConfig`

**Module**: `attractor.agent.session`

```python
@dataclass
class SessionConfig:
    max_turns: int = 0
    max_tool_rounds_per_input: int = 0
    default_command_timeout_ms: int = 10_000
    reasoning_effort: ReasoningEffort | None = None
    enable_loop_detection: bool = True
    max_subagent_depth: int = 1
    model_id: str = ""
    user_instructions: str = ""
    truncation_config: TruncationConfig | None = None
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | `int` | `0` | Maximum LLM turns per `submit()` call. `0` = unlimited. |
| `max_tool_rounds_per_input` | `int` | `0` | Maximum tool-use rounds per input. `0` = unlimited. |
| `default_command_timeout_ms` | `int` | `10_000` | Default shell command timeout in milliseconds. |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` | Model reasoning budget. `None` = provider default. |
| `enable_loop_detection` | `bool` | `True` | Enable `LoopDetector` to emit warnings on repeating tool-call patterns. |
| `max_subagent_depth` | `int` | `1` | Maximum nesting depth for subagents spawned via `spawn_agent`. |
| `model_id` | `str` | `""` | Model identifier passed to the LLM client and embedded in the system prompt. |
| `user_instructions` | `str` | `""` | Additional instructions appended to the system prompt. |
| `truncation_config` | `TruncationConfig \| None` | `None` | Output truncation limits. Uses `TruncationConfig` defaults if `None`. |

---

### `SessionState`

**Module**: `attractor.agent.session`

```python
class SessionState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    CLOSED = "closed"
```

| Value | Description |
|-------|-------------|
| `IDLE` | No active execution. Ready to accept `submit()`. |
| `RUNNING` | Agentic loop is executing. |
| `CLOSED` | Session has been shut down. `submit()` raises `RuntimeError`. |

---

### `AgentLoop`

**Module**: `attractor.agent.loop`

```python
class AgentLoop:
    def __init__(
        self,
        profile: ProviderProfile,
        environment: ExecutionEnvironment,
        registry: ToolRegistry,
        llm_client: LLMClientProtocol,
        emitter: EventEmitter,
        config: LoopConfig,
        loop_detector: LoopDetector,
    ) -> None: ...
```

Executes the agentic tool-use loop. Stateless between runs — all conversation state lives in the `history` list passed in from `Session`.

**Constructor parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile` | `ProviderProfile` | Provider profile for tool definitions and system prompt. |
| `environment` | `ExecutionEnvironment` | Execution environment for tool calls. |
| `registry` | `ToolRegistry` | Tool dispatch registry. |
| `llm_client` | `LLMClientProtocol` | LLM client. |
| `emitter` | `EventEmitter` | Event emitter for streaming events to `Session`. |
| `config` | `LoopConfig` | Loop configuration. |
| `loop_detector` | `LoopDetector` | Shared loop detector instance. |

**Methods**:

#### `run(user_input: str, history: list[Message]) -> None` (async)

Execute the agentic loop for a single user input. Modifies `history` in place.

**Loop termination conditions**:

1. Text-only LLM response (no tool calls) — natural completion.
2. `config.max_turns` reached — emits `TURN_LIMIT` event and returns.
3. `config.max_tool_rounds` reached — emits `TURN_LIMIT` event and returns.
4. LLM call exception — emits `ERROR` event and returns.

#### `queue_steering(message: str) -> None`

Queue a steering message for injection after the current tool round.

---

### `LoopConfig`

**Module**: `attractor.agent.loop`

```python
class LoopConfig:
    def __init__(
        self,
        max_turns: int = 0,
        max_tool_rounds: int = 0,
        enable_loop_detection: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
        default_command_timeout_ms: int = 10_000,
        truncation_config: TruncationConfig | None = None,
        model_id: str = "",
        user_instructions: str = "",
    ) -> None: ...
```

Configuration passed into `AgentLoop` from `Session`.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_turns` | `int` | `0` | Maximum total turns. `0` = unlimited. |
| `max_tool_rounds` | `int` | `0` | Maximum tool-use rounds. `0` = unlimited. |
| `enable_loop_detection` | `bool` | `True` | Enable loop detection. |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` | Reasoning budget. |
| `default_command_timeout_ms` | `int` | `10_000` | Shell timeout. |
| `truncation_config` | `TruncationConfig \| None` | `None` | Truncation limits. |
| `model_id` | `str` | `""` | Model identifier. |
| `user_instructions` | `str` | `""` | Additional system prompt instructions. |

---

### `LLMClientProtocol`

**Module**: `attractor.agent.loop`

```python
@runtime_checkable
class LLMClientProtocol(Protocol):
    async def complete(self, request: Request) -> Response: ...
```

Minimal interface the loop requires from an LLM client.

---

### `ExecutionEnvironment` protocol

**Module**: `attractor.agent.environment`

```python
@runtime_checkable
class ExecutionEnvironment(Protocol):
    async def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> str: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def file_exists(self, path: str) -> bool: ...

    async def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]: ...

    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult: ...

    async def grep(
        self, pattern: str, path: str, options: dict[str, str] | None = None
    ) -> str: ...

    async def glob(self, pattern: str, path: str | None = None) -> list[str]: ...

    async def initialize(self) -> None: ...

    async def cleanup(self) -> None: ...

    def working_directory(self) -> str: ...

    def platform(self) -> str: ...
```

Interface that agent tools use to interact with the outside world. Structural typing — no inheritance required.

**Method details**:

| Method | Description |
|--------|-------------|
| `read_file(path, offset=None, limit=None)` | Read a file. Returns line-numbered content (`"     N\tline"`). `offset` is 1-based. |
| `write_file(path, content)` | Create or overwrite a file. Creates parent directories. |
| `file_exists(path)` | Return `True` if the path exists. |
| `list_directory(path, depth=1)` | Return directory entries up to `depth` levels deep. |
| `exec_command(command, timeout_ms, working_dir, env_vars)` | Execute a shell command. Returns `ExecResult`. |
| `grep(pattern, path, options)` | Regex search returning `"file:line: content"` matches. |
| `glob(pattern, path=None)` | Return file paths matching `pattern`, sorted by modification time descending. |
| `initialize()` | Lifecycle: called before first use. |
| `cleanup()` | Lifecycle: called on `Session.shutdown()`. |
| `working_directory()` | Return the working directory path string. |
| `platform()` | Return the platform string (e.g. `"darwin"`, `"linux"`). |

---

### `DirEntry`

**Module**: `attractor.agent.environment`

```python
@dataclass
class DirEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
```

A single directory entry returned by `list_directory()`.

---

### `ExecResult`

**Module**: `attractor.agent.environment`

```python
@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
```

Result of a shell command execution.

---

### `LocalExecutionEnvironment`

**Module**: `attractor.agent.environment`

```python
class LocalExecutionEnvironment:
    def __init__(self, working_dir: str | None = None) -> None: ...
```

`ExecutionEnvironment` backed by the host filesystem and OS.

**Constructor**:

- `working_dir` (`str | None`): Working directory. Defaults to `Path.cwd()`.

**Behavior details**:

- Relative paths in `read_file`, `write_file`, etc. are resolved against `working_dir`.
- `exec_command` spawns processes with `start_new_session=True`. On timeout, sends SIGTERM to the process group, then SIGKILL after 2 seconds.
- `exec_command` filters sensitive environment variables (matching `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`) from the subprocess environment.
- `grep` uses Python `re` — not the system `grep` binary.
- `glob` results are sorted by `mtime` descending.
- `cleanup()` is a no-op for the local environment.

---

### `AgentEvent`

**Module**: `attractor.agent.events`

```python
@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

A single event from the agent execution pipeline.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `AgentEventType` | The kind of event. |
| `data` | `dict[str, Any]` | Event payload. Shape varies by `type`. |
| `timestamp` | `float` | UNIX epoch when the event was created. |

---

### `AgentEventType`

**Module**: `attractor.agent.events`

```python
class AgentEventType(str, enum.Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
```

**Event data shapes**:

| Event type | `data` keys |
|------------|-------------|
| `SESSION_START` | `{"state": str}` |
| `SESSION_END` | `{"state": str}` |
| `USER_INPUT` | `{"text": str}` |
| `ASSISTANT_TEXT_START` | `{}` |
| `ASSISTANT_TEXT_DELTA` | `{"text": str}` |
| `ASSISTANT_TEXT_END` | `{"text": str}` — full accumulated text |
| `TOOL_CALL_START` | `{"tool_name": str, "tool_call_id": str, "arguments": dict}` |
| `TOOL_CALL_OUTPUT_DELTA` | `{"tool_call_id": str, "output": str}` — truncated |
| `TOOL_CALL_END` | `{"tool_name": str, "tool_call_id": str, "output": str, "full_output": str, "is_error": bool}` |
| `STEERING_INJECTED` | `{"text": str}` |
| `TURN_LIMIT` | `{"turns": int, "limit": int}` |
| `LOOP_DETECTION` | `{"warning": str}` |
| `ERROR` | `{"error": str, "phase": str}` — `phase` is `"llm_call"` or `"session"` |

---

### `EventEmitter`

**Module**: `attractor.agent.events`

```python
class EventEmitter:
    def __init__(self) -> None: ...
```

Async event emitter with iterator-based delivery. Events are placed into an internal `asyncio.Queue` and consumed by async-iterating over the emitter instance.

**Methods**:

#### `emit(event: AgentEvent) -> None`

Enqueue an event for delivery. No-op after `close()`.

#### `close() -> None`

Signal that no more events will be emitted. Enqueues a sentinel that causes `__anext__` to raise `StopAsyncIteration`.

**Usage**:

```python
async for event in emitter:
    print(event.type)
```

---

### `LoopDetector`

**Module**: `attractor.agent.loop_detection`

```python
@dataclass
class LoopDetector:
    _history: list[str] = field(default_factory=list)
```

Tracks tool call history and detects repeating patterns.

**Detection algorithm**:

- Fingerprints each tool call as `sha256("{tool_name}:{arguments_hash}")[:16]`.
- Checks the last `window_size` (default: 10) calls for repeating cycles of length 1, 2, or 3.
- A pattern is a loop if it repeats at least 3 consecutive times.

**Methods**:

#### `record_call(tool_name: str, arguments_hash: str) -> None`

Record a tool call fingerprint. `arguments_hash` is a pre-computed hash of the arguments dict.

#### `check_for_loops(window_size: int = 10) -> str | None`

Check for repeating cycles. Returns a warning message string if detected, or `None`.

#### `reset() -> None`

Clear all recorded history.

---

### `TruncationConfig`

**Module**: `attractor.agent.truncation`

```python
@dataclass
class TruncationConfig:
    char_limits: dict[str, int] = field(default_factory=lambda: { ... })
    line_limits: dict[str, int] = field(default_factory=lambda: { ... })
```

Per-tool truncation limits. Setting a limit to `0` disables that stage for the tool.

**Default character limits (Stage 1)**:

| Tool | Character limit |
|------|----------------|
| `read_file` | 50,000 |
| `shell` | 30,000 |
| `grep` | 20,000 |
| `glob` | 20,000 |
| `edit_file` | 10,000 |
| `apply_patch` | 10,000 |
| `write_file` | 1,000 |

**Default line limits (Stage 2)**:

| Tool | Line limit |
|------|-----------|
| `shell` | 256 |
| `grep` | 200 |
| `glob` | 500 |

**Truncation format**: When truncation occurs, the middle is replaced with:

```
[WARNING: Tool output was truncated. N characters/lines removed from middle of output]
```

Head and tail each receive half the limit.

#### `truncate_output()`

```python
def truncate_output(
    tool_name: str,
    output: str,
    config: TruncationConfig | None = None,
) -> tuple[str, str]
```

Apply the two-stage truncation pipeline.

**Returns**: `(truncated, full_original)` — `truncated` is sent to the LLM; `full_original` is the unmodified output stored in `TOOL_CALL_END`.

---

### `ProviderProfile` ABC

**Module**: `attractor.agent.profiles.base`

```python
class ProviderProfile(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def tool_definitions(self) -> list[ToolDefinition]: ...

    @property
    @abstractmethod
    def system_prompt_template(self) -> str: ...

    @property
    @abstractmethod
    def context_window_size(self) -> int: ...

    def get_tools(self) -> list[ToolDefinition]: ...

    def format_system_prompt(self, **kwargs: str) -> str: ...
```

Provider-specific configuration for tools and system prompts.

**Abstract properties**:

| Property | Return type | Description |
|----------|-------------|-------------|
| `provider_name` | `str` | Short provider identifier (`"anthropic"`, `"openai"`, `"google"`). |
| `tool_definitions` | `list[ToolDefinition]` | Tool definitions exposed to the model. |
| `system_prompt_template` | `str` | Template string with `{placeholder}` variables. |
| `context_window_size` | `int` | Maximum context window in tokens. |

**Concrete methods**:

#### `get_tools() -> list[ToolDefinition]`

Returns `list(self.tool_definitions)`.

#### `format_system_prompt(**kwargs: str) -> str`

Replace `{key}` placeholders in `system_prompt_template` with provided values.

**Common variables**: `working_dir`, `platform`, `date`, `model_id`, `git_branch`, `git_status`, `project_docs`, `user_instructions`.

---

### Provider profiles

#### `AnthropicProfile`

**Module**: `attractor.agent.profiles.anthropic_profile`

- `provider_name`: `"anthropic"`
- `context_window_size`: `200_000`
- **Tools**: `edit_file`, `read_file`, `write_file`, `shell`, `grep`, `glob`, `spawn_agent`
- **Edit tool**: `edit_file` (search-and-replace)

#### `OpenAIProfile`

**Module**: `attractor.agent.profiles.openai_profile`

- `provider_name`: `"openai"`
- `context_window_size`: `128_000`
- **Tools**: `apply_patch`, `read_file`, `write_file`, `shell`, `grep`, `glob`, `spawn_agent`
- **Edit tool**: `apply_patch` (v4a unified diff format)

#### `GeminiProfile`

**Module**: `attractor.agent.profiles.gemini_profile`

- `provider_name`: `"google"`
- `context_window_size`: `1_000_000`
- **Tools**: `edit_file`, `read_file`, `write_file`, `shell`, `grep`, `glob`, `list_dir`, `spawn_agent`
- **Additional tool**: `list_dir` — list directory contents with file sizes.

---

### `build_system_prompt()`

**Module**: `attractor.agent.prompts`

```python
def build_system_prompt(
    profile: ProviderProfile,
    environment: ExecutionEnvironment,
    model_id: str = "",
    user_instructions: str = "",
) -> str
```

Construct the full system prompt for an agent session.

**Layers** (later overrides earlier):

1. Provider base prompt with environment context.
2. Project documentation discovered from the working directory.
3. User instruction overrides.

**Project documentation discovery** (`discover_project_docs()`):

- Walks from the current working directory up to the git repository root.
- Loads `AGENTS.md` and provider-specific files (`CLAUDE.md` for Anthropic, `.codex/instructions.md` for OpenAI, `GEMINI.md` for Google).
- Enforces a 32 KB total budget. Files exceeding the budget are truncated with a note.

---

### `ToolRegistry`

**Module**: `attractor.agent.tools.registry`

```python
class ToolRegistry:
    def __init__(self) -> None: ...
```

Maps tool names to async handler callables and their `ToolDefinition`s.

**Methods**:

#### `register(name: str, handler: ToolHandler, definition: ToolDefinition) -> None`

Register a tool. Overwrites any existing registration for `name`.

#### `dispatch(name: str, arguments: dict[str, Any], environment: ExecutionEnvironment) -> ToolResult` (async)

Look up and execute a tool by name.

- Returns an error `ToolResult` for unknown tools rather than raising.
- Returns an error `ToolResult` if the handler raises an exception.

#### `definitions() -> list[ToolDefinition]`

Return all registered tool definitions.

#### `has_tool(name: str) -> bool`

Return `True` if `name` is registered.

#### `tool_names() -> list[str]`

Return all registered tool name strings.

---

### `ToolResult`

**Module**: `attractor.agent.tools.registry`

```python
@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    full_output: str = ""
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | `str` | — | Text sent back to the LLM (possibly truncated). |
| `is_error` | `bool` | `False` | `True` if the tool produced an error. |
| `full_output` | `str` | `""` | The untruncated original output. |

---

### Tool definitions

All tools accept `(arguments: dict[str, Any], environment: ExecutionEnvironment) -> ToolResult` (async).

#### `read_file`

**Module**: `attractor.agent.tools.core_tools`

Read a file and return line-numbered content.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | `string` | Yes | Absolute or relative path to the file. |
| `offset` | `integer` | No | 1-based line number to start reading from. |
| `limit` | `integer` | No | Maximum number of lines to read. |

**Returns**: Line-numbered content as `"     N\tline content"`. Error if file not found.

**Truncation**: 50,000 characters (Stage 1 only).

---

#### `write_file`

**Module**: `attractor.agent.tools.core_tools`

Create or overwrite a file with given content.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | `string` | Yes | Absolute or relative path for the file. |
| `content` | `string` | Yes | The full content to write. |

**Returns**: `"Successfully wrote to {path}"`. Error on write failure.

**Truncation**: 1,000 characters (Stage 1 only).

---

#### `edit_file`

**Module**: `attractor.agent.tools.core_tools`

Make a search-and-replace edit in a file. Used by `AnthropicProfile` and `GeminiProfile`.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | `string` | Yes | Path to the file to edit. |
| `old_string` | `string` | Yes | The exact text to find. Must appear exactly once in the file. |
| `new_string` | `string` | Yes | The replacement text. |

**Returns**: `"Successfully edited {path}"`. Errors if file not found, `old_string` not found, or `old_string` appears more than once.

**Truncation**: 10,000 characters (Stage 1 only).

---

#### `shell`

**Module**: `attractor.agent.tools.core_tools`

Execute a shell command.

**Parameters**:

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `command` | `string` | Yes | — | The shell command to execute. |
| `timeout_ms` | `integer` | No | `10000` | Timeout in milliseconds. |

**Returns**: Combined stdout/stderr output. `is_error=True` if exit code != 0 or timed out.

**Truncation**: 30,000 characters (Stage 1), 256 lines (Stage 2).

---

#### `grep`

**Module**: `attractor.agent.tools.core_tools`

Search for a regex pattern across files.

**Parameters**:

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `pattern` | `string` | Yes | — | Regular expression pattern. |
| `path` | `string` | No | working directory | File or directory to search in. |
| `include` | `string` | No | `None` | Glob pattern to filter files (e.g. `*.py`). |
| `max_results` | `integer` | No | `100` | Maximum number of results. |

**Returns**: Newline-separated `"file:line: content"` matches, or `"No matches found."`.

**Truncation**: 20,000 characters (Stage 1), 200 lines (Stage 2).

---

#### `glob`

**Module**: `attractor.agent.tools.core_tools`

Find files matching a glob pattern, sorted by modification time descending.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `pattern` | `string` | Yes | Glob pattern (e.g. `**/*.py`). |
| `path` | `string` | No | Base directory for the search. Defaults to working directory. |

**Returns**: Newline-separated file paths, or `"No files matched."`.

**Truncation**: 20,000 characters (Stage 1), 500 lines (Stage 2).

---

#### `apply_patch`

**Module**: `attractor.agent.tools.apply_patch`

Apply a v4a-format patch to create, modify, delete, or move files. Used by `OpenAIProfile`.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `patch` | `string` | Yes | The patch text in v4a unified diff format. |

**Supported operations**:

| Operation | Triggered by |
|-----------|-------------|
| `ADD_FILE` | `--- /dev/null` followed by `+++ b/path` |
| `DELETE_FILE` | `+++ /dev/null` following `--- a/path` |
| `UPDATE_FILE` | `diff --git a/path b/path` with same old and new path |
| `MOVE_FILE` | `diff --git a/old b/new` with different paths |

**Returns**: `"Patch applied successfully to: {files}"`. Error listing failed operations if any fail.

**Truncation**: 10,000 characters (Stage 1 only).

---

#### `spawn_agent`

**Module**: `attractor.agent.tools.subagent`

Spawn a subagent to work on a delegated task.

**Parameters**:

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `task` | `string` | Yes | — | Description of the task for the subagent. |
| `working_dir` | `string` | No | — | Working directory for the subagent. |
| `model` | `string` | No | — | Model to use for the subagent. |
| `max_turns` | `integer` | No | — | Maximum turns for the subagent. |

**Returns**: Message with the generated `agent_id` (format: `subagent_{8 hex chars}`). Use `wait(agent_id=...)` to retrieve results.

---

#### `send_input`

**Module**: `attractor.agent.tools.subagent`

Send a follow-up message to a running subagent.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_id` | `string` | Yes | The subagent ID. |
| `message` | `string` | Yes | The message to send. |

**Returns**: Confirmation string. Error if subagent not found or closed.

---

#### `wait`

**Module**: `attractor.agent.tools.subagent`

Wait for a subagent to complete and return its result.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_id` | `string` | Yes | The subagent ID to wait on. |

**Returns**: The subagent's result string. Error if not found or if the task failed.

---

#### `close_agent`

**Module**: `attractor.agent.tools.subagent`

Close a subagent and free its resources.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_id` | `string` | Yes | The subagent ID to close. |

**Returns**: Confirmation string. Cancels any running task.

---

## 4. LLM Client API Reference

**Module prefix**: `attractor.llm`

---

### `LLMClient`

**Module**: `attractor.llm.client`

```python
class LLMClient:
    def __init__(
        self,
        adapters: list[Any] | None = None,
        middleware: list[Middleware] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...
```

Provider-agnostic LLM client with middleware and tool-loop support. Routes requests to the appropriate provider adapter based on model name.

**Constructor parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapters` | `list[Any] \| None` | `None` | Provider adapters. If `None`, loads all available adapters via `_default_adapters()`. |
| `middleware` | `list[Middleware] \| None` | `None` | Middleware pipeline. Applied in order on request, in reverse order on response. |
| `retry_policy` | `RetryPolicy \| None` | `None` | Retry configuration. Defaults to `RetryPolicy()`. |

**Adapter loading**: `_default_adapters()` tries to import `OpenAIAdapter`, `AnthropicAdapter`, and `GeminiAdapter` in that order. Adapters whose provider SDK is not installed are silently skipped.

**Methods**:

#### `detect_provider(model: str) -> ProviderAdapter`

Return the first adapter that claims the model string.

**Raises**: `ValueError` if no adapter claims the model.

#### `complete(request: Request) -> Response` (async)

Send a completion request, applying middleware and retries.

1. Detect provider via model name.
2. Apply `before_request` middleware (in order).
3. Call adapter with retry logic.
4. Apply `after_response` middleware (in reverse order).

#### `stream(request: Request) -> AsyncIterator[StreamEvent]` (async generator)

Send a streaming request, applying `before_request` middleware only. Yields `StreamEvent` objects.

#### `generate(prompt, model, tools=None, tool_executor=None, max_tool_rounds=10, **kwargs) -> Response` (async)

```python
async def generate(
    self,
    prompt: str | list[Message],
    model: str,
    tools: list[ToolDefinition] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_tool_rounds: int = 10,
    **kwargs: Any,
) -> Response
```

High-level API with automatic tool execution loop.

- `prompt`: String (converted to a user message) or list of `Message` objects.
- `tool_executor`: Async callable `(ToolCallContent) -> str`. All tool calls in a single round are executed concurrently via `asyncio.gather`.
- `max_tool_rounds`: Maximum tool-use round trips. Default: `10`.
- `**kwargs`: Additional `Request` fields (`temperature`, `max_tokens`, `system_prompt`, etc.).

#### `stream_generate(prompt, model, tools=None, **kwargs) -> AsyncIterator[StreamEvent]` (async generator)

Streaming version of `generate`. Single round only — no tool loop.

#### `generate_object(prompt, model, schema, schema_name="response", strict=True, **kwargs) -> dict[str, Any]` (async)

Generate a structured JSON object validated against a JSON Schema.

- `schema`: JSON Schema dict describing the expected output shape.
- `schema_name`: Name for the schema (used by some providers). Default: `"response"`.
- `strict`: Whether to enforce strict schema adherence. Default: `True`.

**Returns**: Parsed JSON as a Python `dict`.

**Raises**: `ValueError` if the model output cannot be parsed as valid JSON.

---

### `Request`

**Module**: `attractor.llm.models`

```python
@dataclass
class Request:
    messages: list[Message] = field(default_factory=list)
    model: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    reasoning_effort: ReasoningEffort | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

A request to an LLM provider.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[Message]` | `[]` | Conversation history. |
| `model` | `str` | `""` | Model identifier (e.g. `"gpt-4o"`, `"claude-opus-4-6"`). |
| `tools` | `list[ToolDefinition]` | `[]` | Tool definitions available to the model. |
| `system_prompt` | `str` | `""` | System instructions. |
| `temperature` | `float \| None` | `None` | Sampling temperature. |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens. Anthropic adapter defaults to `4096` if `None`. |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability. |
| `stop_sequences` | `list[str]` | `[]` | Stop sequences. |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` | Model reasoning budget. |
| `response_format` | `dict[str, Any] \| None` | `None` | Structured output format specification. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata passed through middleware. |

---

### `Response`

**Module**: `attractor.llm.models`

```python
@dataclass
class Response:
    message: Message = field(default_factory=lambda: Message(role=Role.ASSISTANT))
    model: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_response_id: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

A response from an LLM provider.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `Message` | empty assistant | The assistant message containing text and/or tool calls. |
| `model` | `str` | `""` | Model identifier as returned by the provider. |
| `finish_reason` | `FinishReason` | `STOP` | Why the model stopped generating. |
| `usage` | `TokenUsage` | empty | Token consumption details. |
| `provider_response_id` | `str` | `""` | Provider-assigned response identifier. |
| `latency_ms` | `float` | `0.0` | Round-trip time in milliseconds. |
| `metadata` | `dict[str, Any]` | `{}` | Provider-specific metadata. |

---

### `Message`

**Module**: `attractor.llm.models`

```python
@dataclass
class Message:
    role: Role
    content: list[ContentPart] = field(default_factory=list)
```

A single message in a conversation.

**Static constructors**:

| Method | Returns |
|--------|---------|
| `Message.system(text: str)` | `Message(role=SYSTEM, content=[TextContent(text=text)])` |
| `Message.user(text: str)` | `Message(role=USER, content=[TextContent(text=text)])` |
| `Message.assistant(text: str)` | `Message(role=ASSISTANT, content=[TextContent(text=text)])` |
| `Message.tool_result(tool_call_id, content, is_error=False)` | `Message(role=TOOL, content=[ToolResultContent(...)])` |

**Methods**:

#### `text() -> str`

Extract and concatenate text from all `TextContent` parts.

#### `tool_calls() -> list[ToolCallContent]`

Extract all `ToolCallContent` parts.

#### `has_tool_calls() -> bool`

Return `True` if any `ToolCallContent` part exists.

---

### Content types

All content types are dataclasses with a `kind: ContentKind` field set automatically.

#### `TextContent`

```python
@dataclass
class TextContent:
    kind: ContentKind  # = ContentKind.TEXT
    text: str = ""
```

#### `ImageContent`

```python
@dataclass
class ImageContent:
    kind: ContentKind  # = ContentKind.IMAGE
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "image/png"
```

#### `AudioContent`

```python
@dataclass
class AudioContent:
    kind: ContentKind  # = ContentKind.AUDIO
    base64_data: str = ""
    media_type: str = "audio/wav"
```

#### `DocumentContent`

```python
@dataclass
class DocumentContent:
    kind: ContentKind  # = ContentKind.DOCUMENT
    base64_data: str = ""
    media_type: str = "application/pdf"
```

#### `ToolCallContent`

```python
@dataclass
class ToolCallContent:
    kind: ContentKind  # = ContentKind.TOOL_CALL
    tool_call_id: str  # default: "call_{12 hex chars}"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_json: str = ""
```

#### `ToolResultContent`

```python
@dataclass
class ToolResultContent:
    kind: ContentKind  # = ContentKind.TOOL_RESULT
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False
```

#### `ThinkingContent`

```python
@dataclass
class ThinkingContent:
    kind: ContentKind  # = ContentKind.THINKING
    text: str = ""
```

Model reasoning/thinking content (extended thinking). Anthropic-specific.

#### `RedactedThinkingContent`

```python
@dataclass
class RedactedThinkingContent:
    kind: ContentKind  # = ContentKind.REDACTED_THINKING
    data: str = ""
```

Redacted model reasoning. Anthropic-specific.

**`ContentPart` union type**:

```python
ContentPart = (
    TextContent | ImageContent | AudioContent | DocumentContent
    | ToolCallContent | ToolResultContent
    | ThinkingContent | RedactedThinkingContent
)
```

---

### `ToolDefinition`

**Module**: `attractor.llm.models`

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
```

Definition of a tool that can be invoked by the model.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Tool name. Must match what the model is told to call. |
| `description` | `str` | — | Human-readable description for the model. |
| `parameters` | `dict[str, Any]` | `{}` | JSON Schema object describing tool parameters. |
| `strict` | `bool` | `False` | Enforce strict schema adherence (used by OpenAI). |

**Methods**:

#### `to_json_schema() -> dict[str, Any]`

Convert to JSON Schema format: `{"type": "object", "properties": ..., "required": [...]}`.

---

### `ToolParameter`

**Module**: `attractor.llm.models`

```python
@dataclass
class ToolParameter:
    name: str
    type: str
    description: str = ""
    required: bool = False
    enum: list[str] | None = None
    default: Any = None
```

A single parameter in a tool's input schema. Utility dataclass — not used directly in API calls (parameters are passed as raw dicts in `ToolDefinition.parameters`).

---

### `TokenUsage`

**Module**: `attractor.llm.models`

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
```

Token consumption details.

**Fields**:

| Field | Description |
|-------|-------------|
| `input_tokens` | Tokens in the request (prompt). |
| `output_tokens` | Tokens in the response. |
| `reasoning_tokens` | Tokens used for extended thinking/reasoning (if applicable). |
| `cache_read_tokens` | Tokens served from prompt cache. |
| `cache_write_tokens` | Tokens written to prompt cache. |

**Property**:

#### `total_tokens -> int`

`input_tokens + output_tokens`.

---

### `RetryPolicy`

**Module**: `attractor.llm.models`

```python
@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 60.0
```

Configuration for request retry behavior with exponential backoff.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `2` | Maximum number of retry attempts after the initial failure. |
| `base_delay_seconds` | `float` | `1.0` | Base delay before the first retry. |
| `multiplier` | `float` | `2.0` | Backoff multiplier applied per attempt. |
| `max_delay_seconds` | `float` | `60.0` | Cap on any single delay. |

**Methods**:

#### `delay_for_attempt(attempt: int) -> float`

Return the delay in seconds for the given attempt number (0-indexed).

Formula: `min(base_delay_seconds * (multiplier ** attempt), max_delay_seconds)`

**Example delays** (defaults):

| Attempt | Delay |
|---------|-------|
| 0 | 1.0 s |
| 1 | 2.0 s |
| 2 | 4.0 s |

---

### `StreamEvent`

**Module**: `attractor.llm.models`

```python
@dataclass
class StreamEvent:
    type: StreamEventType
    text: str = ""
    tool_call: ToolCallContent | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

A single event in a streaming response.

**Fields**:

| Field | Type | Present for |
|-------|------|-------------|
| `type` | `StreamEventType` | All events |
| `text` | `str` | `TEXT_DELTA`, `REASONING_DELTA`, `TOOL_CALL_DELTA` |
| `tool_call` | `ToolCallContent \| None` | `TOOL_CALL_START`, `TOOL_CALL_DELTA`, `TOOL_CALL_END` |
| `finish_reason` | `FinishReason \| None` | `FINISH` |
| `usage` | `TokenUsage \| None` | `FINISH`, `STREAM_START` |
| `error` | `str \| None` | `ERROR` |
| `metadata` | `dict[str, Any]` | `FINISH` — contains `latency_ms` |

---

### `StreamEventType`

**Module**: `attractor.llm.models`

```python
class StreamEventType(str, enum.Enum):
    STREAM_START = "stream_start"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"
    ERROR = "error"
```

| Value | Description |
|-------|-------------|
| `STREAM_START` | First event in a stream. May carry input token `usage`. |
| `TEXT_DELTA` | Incremental text output chunk. |
| `REASONING_DELTA` | Incremental reasoning/thinking chunk. |
| `TOOL_CALL_START` | A tool call has begun. `tool_call` carries initial metadata. |
| `TOOL_CALL_DELTA` | Incremental tool call argument chunk. `text` is partial JSON. |
| `TOOL_CALL_END` | Tool call arguments complete. `tool_call` has fully parsed `arguments`. |
| `FINISH` | Stream complete. Carries `finish_reason`, `usage`, `metadata.latency_ms`. |
| `ERROR` | Stream error. Carries `error` string. |

---

### `StreamCollector`

**Module**: `attractor.llm.streaming`

```python
@dataclass
class StreamCollector:
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallContent] = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    metadata: dict = field(default_factory=dict)
```

Accumulates `StreamEvent`s into a complete `Response`.

**Methods**:

#### `process_event(event: StreamEvent) -> None`

Process a single stream event, updating internal state.

#### `to_response() -> Response`

Assemble all accumulated events into a complete `Response`.

#### `collect(stream) -> Response` (async)

Consume an entire async stream and return the assembled `Response`.

**Example**:

```python
collector = StreamCollector()
async for event in client.stream(request):
    collector.process_event(event)
response = collector.to_response()
```

---

### `ProviderAdapter` protocol

**Module**: `attractor.llm.adapters.base`

```python
@runtime_checkable
class ProviderAdapter(Protocol):
    def provider_name(self) -> str: ...
    def detect_model(self, model: str) -> bool: ...
    async def complete(self, request: Request) -> Response: ...
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]: ...
```

Protocol that all provider adapters must satisfy.

**Methods**:

| Method | Description |
|--------|-------------|
| `provider_name()` | Return provider identifier string. |
| `detect_model(model)` | Return `True` if this adapter handles the given model string. |
| `complete(request)` | Send a non-streaming completion request. Returns `Response`. |
| `stream(request)` | Send a streaming request. Yields `StreamEvent` objects. |

---

### Provider adapters

#### `AnthropicAdapter`

**Module**: `attractor.llm.adapters.anthropic_adapter`

```python
class AnthropicAdapter:
    def __init__(self, api_key: str | None = None) -> None: ...
```

Adapter for the Anthropic Messages API.

- **Model detection**: `model.startswith("claude-")`
- **API key**: `api_key` or `ANTHROPIC_API_KEY` environment variable.
- **`max_tokens`**: Defaults to `4096` if not specified.
- **Message alternation**: Inserts synthetic `"..."` placeholder messages to enforce strict user/assistant alternation required by the Anthropic API.
- **Reasoning effort mapping**:

| `ReasoningEffort` | Budget tokens |
|-------------------|---------------|
| `LOW` | 2,048 |
| `MEDIUM` | 8,192 |
| `HIGH` | 32,768 |

- **Finish reason mapping**:

| Anthropic `stop_reason` | `FinishReason` |
|-------------------------|----------------|
| `"end_turn"` | `STOP` |
| `"tool_use"` | `TOOL_USE` |
| `"max_tokens"` | `LENGTH` |
| `"stop_sequence"` | `STOP` |

---

#### `OpenAIAdapter`

**Module**: `attractor.llm.adapters.openai_adapter`

```python
class OpenAIAdapter:
    def __init__(self, api_key: str | None = None) -> None: ...
```

Adapter for the OpenAI Responses API (`client.responses.create()`).

- **Model detection**: `model.startswith(("gpt-", "o1", "o3", "o4"))`
- **API key**: `api_key` or `OPENAI_API_KEY` environment variable.
- **Uses**: OpenAI Responses API (not `chat.completions`) — returns typed output items.
- **`reasoning_effort`**: Maps to `{"reasoning": {"effort": value}}`.
- **`response_format`**: Maps to `{"text": {"format": value}}`.
- **`max_tokens`**: Maps to `max_output_tokens`.

---

#### `GeminiAdapter`

**Module**: `attractor.llm.adapters.gemini_adapter`

```python
class GeminiAdapter:
    def __init__(self, api_key: str | None = None) -> None: ...
```

Adapter for the Google Gemini API (`google-genai`).

- **Model detection**: `model.startswith("gemini-")`
- **API key**: `api_key` or `GOOGLE_API_KEY` environment variable.
- **System prompt handling**: Injected as a user/model exchange (`"[System Instructions]\n{prompt}"` → `"Understood."`), since Gemini does not have a first-class system role in the contents array.
- **Tool call IDs**: Synthesized as `call_{12 hex chars}` (Gemini does not return IDs).
- **`max_tokens`**: Maps to `max_output_tokens` in config.
- **`reasoning_effort`**: Not mapped (Gemini does not support this parameter).

**Finish reason mapping**:

| Gemini finish reason | `FinishReason` |
|----------------------|----------------|
| `"STOP"` | `STOP` |
| `"MAX_TOKENS"` | `LENGTH` |
| `"SAFETY"` | `CONTENT_FILTER` |
| `"RECITATION"` | `CONTENT_FILTER` |

---

### `Middleware` protocol

**Module**: `attractor.llm.middleware`

```python
@runtime_checkable
class Middleware(Protocol):
    async def before_request(self, request: Request) -> Request: ...
    async def after_response(self, response: Response) -> Response: ...
```

Protocol for request/response middleware.

**Pipeline behavior**:

- `before_request` is called in registration order.
- `after_response` is called in reverse registration order.

---

### Built-in middleware

#### `LoggingMiddleware`

**Module**: `attractor.llm.middleware`

```python
class LoggingMiddleware:
    def __init__(self, log_level: int = logging.INFO) -> None: ...
```

Logs request metadata (model, message count, tool count) and response latency/token usage at the specified log level.

#### `TokenTrackingMiddleware`

**Module**: `attractor.llm.middleware`

```python
@dataclass
class TokenTrackingMiddleware:
    total_usage: TokenUsage = field(default_factory=TokenUsage)
```

Accumulates token usage across all calls in `total_usage`.

#### `RetryMiddleware`

**Module**: `attractor.llm.middleware`

```python
class RetryMiddleware:
    def __init__(self, policy: RetryPolicy | None = None) -> None: ...
    last_attempt: int
```

Tracking middleware around retry behavior. Note: the actual retry loop is in `LLMClient` — this middleware provides timing/logging context.

---

### Enums

#### `Role`

**Module**: `attractor.llm.models`

```python
class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"
```

| Value | Description |
|-------|-------------|
| `SYSTEM` | System instructions (not included in Anthropic/Gemini message lists — extracted to `system` parameter). |
| `USER` | Human turn. |
| `ASSISTANT` | Model turn. |
| `TOOL` | Tool result message. |
| `DEVELOPER` | Developer-role message (OpenAI-specific). |

#### `ContentKind`

**Module**: `attractor.llm.models`

```python
class ContentKind(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
```

Discriminator for the `ContentPart` tagged union.

#### `FinishReason`

**Module**: `attractor.llm.models`

```python
class FinishReason(str, enum.Enum):
    STOP = "stop"
    TOOL_USE = "tool_use"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
```

| Value | Description |
|-------|-------------|
| `STOP` | Natural completion (model chose to stop). |
| `TOOL_USE` | Model returned tool calls. |
| `LENGTH` | `max_tokens` reached. |
| `CONTENT_FILTER` | Output blocked by safety filters. |
| `ERROR` | Error during generation. |

#### `ReasoningEffort`

**Module**: `attractor.llm.models`

```python
class ReasoningEffort(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

Model reasoning/thinking budget level.

| Value | Anthropic budget tokens | OpenAI `reasoning.effort` |
|-------|------------------------|--------------------------|
| `LOW` | 2,048 | `"low"` |
| `MEDIUM` | 8,192 | `"medium"` |
| `HIGH` | 32,768 | `"high"` |

#### `StreamEventType`

See [StreamEventType](#streameventtype) above.

---

### `ModelInfo`

**Module**: `attractor.llm.models`

```python
@dataclass
class ModelInfo:
    model_id: str
    provider: str
    display_name: str = ""
    context_window: int = 128_000
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
```

Metadata about a specific model. Informational — not used by the LLM client routing logic.

---

## Cross-References

- New to Attractor? Start with the [Tutorial](../tutorials/getting-started.md)
- For practical tasks, see [How-to Guides](../how-to/common-tasks.md)
- To understand design decisions, read [Explanation](../explanation/architecture.md)
