# Run Directory Structure — Spec Compliance Gap

## Status: Not Implemented

This document describes the run directory layout required by the attractor
specification. The engine currently uses a flat `checkpoint_dir` for checkpoint
files only — no per-node stage directories, manifest, or artifact storage exist.

---

## Overview

Each pipeline execution creates a run directory tree under `{logs_root}`. This
directory serves as the working area for the run, containing per-node output
files, checkpoints, and large artifacts.

---

## Directory Layout

```
{logs_root}/
    manifest.json                    # Pipeline metadata
    checkpoint.json                  # Latest execution state snapshot

    {node_id}/                       # One directory per executed node
        status.json                  # Handler outcome and routing info
        prompt.md                    # Rendered prompt sent to LLM (codergen only)
        response.md                  # Raw LLM response text (codergen only)

    artifacts/                       # Large output storage
        {artifact_id}.json           # File-backed artifacts (>100KB threshold)
```

### Example

For a pipeline with nodes `start`, `code`, `review`, `exit`:

```
.attractor/runs/2026-02-21T18-30-00/
    manifest.json
    checkpoint.json

    start/
        status.json

    code/
        status.json
        prompt.md
        response.md

    review/
        status.json
        prompt.md
        response.md

    exit/
        status.json

    artifacts/
        a1b2c3d4.json
```

---

## File Specifications

### manifest.json

Written once at pipeline start. Updated at completion.

```json
{
    "pipeline_name": "my_pipeline",
    "goal": "Fix the authentication bug in login flow",
    "start_time": "2026-02-21T18:30:00Z",
    "end_time": "2026-02-21T18:45:12Z",
    "status": "completed",
    "start_node": "start",
    "node_count": 4,
    "model": "claude-sonnet-4-20250514"
}
```

| Field | Type | Written | Description |
|-------|------|---------|-------------|
| `pipeline_name` | `string` | Start | Graph name from the DOT file |
| `goal` | `string` | Start | Top-level goal from graph attributes |
| `start_time` | `string` (ISO 8601) | Start | When execution began |
| `end_time` | `string \| null` | Completion | When execution ended (null if in progress) |
| `status` | `string` | Start + completion | `"running"`, `"completed"`, `"failed"`, `"resumed"` |
| `start_node` | `string` | Start | Name of the designated start node |
| `node_count` | `int` | Start | Total number of nodes in the pipeline |
| `model` | `string \| null` | Start | Default model from stylesheet or CLI |

### checkpoint.json

Written after every node execution. This replaces the current behavior of
writing timestamped files to `checkpoint_dir`.

The schema matches the existing `Checkpoint.to_dict()` output:

```json
{
    "pipeline_name": "my_pipeline",
    "current_node": "review",
    "context": { "code.output": "..." },
    "completed_nodes": ["start", "code"],
    "timestamp": 1771699200.0,
    "node_retries": { "code": 1 },
    "logs": [
        { "node": "start", "outcome": "success", "timestamp": 1771699100.0 }
    ]
}
```

Only one `checkpoint.json` exists at a time (latest state). The engine
overwrites it after each node.

### {node_id}/status.json

See [status-file-contract.md](status-file-contract.md) for full details.

### {node_id}/prompt.md

Written by codergen (LLM-backed) handlers before the LLM call.

Contains the fully rendered prompt after template interpolation of context
variables. Useful for auditing and debugging prompt construction.

```markdown
# Task

Fix the authentication bug in the login flow.

## Context

- Previous stage: code_analysis
- Outcome: success
- Files identified: src/auth/login.py, src/auth/session.py
```

### {node_id}/response.md

Written by codergen handlers after the LLM responds.

Contains the raw LLM response text (not including tool calls or thinking
blocks — just the final text output).

### artifacts/{artifact_id}.json

Large outputs (>100KB) are written here instead of being stored inline in the
pipeline context. The context stores a reference:

```json
{
    "type": "artifact_ref",
    "artifact_id": "a1b2c3d4",
    "size_bytes": 245000,
    "summary": "Generated API client code (3 files)"
}
```

---

## Implementation Requirements

### 1. Run Directory Creation

The engine should create the run directory at pipeline start:

```python
import datetime

def _create_run_dir(self, base_dir: str, pipeline_name: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(base_dir) / "runs" / f"{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
```

### 2. Manifest Writing

Write `manifest.json` at pipeline start and update at completion:

```python
def _write_manifest(self, run_dir: Path, pipeline: Pipeline, status: str) -> None:
    manifest = {
        "pipeline_name": pipeline.name,
        "goal": pipeline.goal,
        "start_time": self._start_time.isoformat(),
        "end_time": datetime.datetime.now().isoformat() if status != "running" else None,
        "status": status,
        "start_node": pipeline.start_node,
        "node_count": len(pipeline.nodes),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

### 3. Stage Directory Creation

Before each handler dispatch:

```python
stage_dir = run_dir / node.name
stage_dir.mkdir(parents=True, exist_ok=True)
```

### 4. Checkpoint Location Change

Move from timestamped files in `checkpoint_dir` to a single `checkpoint.json`
in the run directory:

```python
# Current: checkpoint_dir / f"checkpoint_{timestamp_ms}.json"
# New:     run_dir / "checkpoint.json"
```

The `resume` command should accept a run directory path and find
`checkpoint.json` within it.

### 5. Artifact Storage

Add a threshold check when storing context values:

```python
ARTIFACT_THRESHOLD = 100 * 1024  # 100KB

def _maybe_externalize(self, run_dir: Path, key: str, value: Any) -> Any:
    serialized = json.dumps(value)
    if len(serialized) > ARTIFACT_THRESHOLD:
        artifact_id = hashlib.sha256(serialized.encode()).hexdigest()[:8]
        artifact_path = run_dir / "artifacts" / f"{artifact_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(serialized)
        return {
            "type": "artifact_ref",
            "artifact_id": artifact_id,
            "size_bytes": len(serialized),
            "summary": f"Externalized value for '{key}'"
        }
    return value
```

### 6. Handler Interface Update

Handlers need access to their stage directory. Options:

**Option A: Via context** (minimal interface change):
```python
context.set("_logs_root", str(run_dir))
context.set("_stage_dir", str(stage_dir))
```

**Option B: Via handler parameter** (cleaner but breaking):
```python
class NodeHandler(Protocol):
    async def execute(
        self, node: PipelineNode, context: PipelineContext,
        stage_dir: Path | None = None
    ) -> NodeResult: ...
```

Option A is recommended for backward compatibility.

---

## Migration Path

The current checkpoint system (`checkpoint_dir` with timestamped files) should
be preserved alongside the new run directory for backward compatibility:

1. If `logs_root` / run directory is configured, use the new layout
2. Otherwise, fall back to the existing `checkpoint_dir` behavior
3. The `resume` command should detect which format is in use

### Files to Modify

| File | Changes |
|------|---------|
| `engine.py` | Run dir creation, manifest writing, stage dir setup, checkpoint location |
| `state.py` | Update checkpoint save/load for new location |
| `handlers.py` | CodergenHandler writes prompt.md/response.md |
| `cli.py` | Add `--logs-root` option, update resume to find run dirs |
| `models.py` | Add artifact ref type (optional) |

---

## Testing Requirements

1. **Run directory created at pipeline start** — verify structure
2. **Manifest written and updated** — verify content at start and completion
3. **Stage directories created per node** — verify before handler dispatch
4. **Checkpoint written to run directory** — verify single file, overwritten
5. **Prompt/response files written by codergen** — verify content matches
6. **Artifact externalization** — verify threshold, file content, context ref
7. **Resume from run directory** — verify checkpoint discovery
8. **Backward compatibility** — verify old checkpoint_dir still works
