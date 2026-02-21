# CLAUDE.md — Attractor

Attractor is a non-interactive coding agent for software factories. It orchestrates LLM-driven workflows defined as GraphViz DOT pipelines, executing them through an autonomous agent with tool-use capabilities.

## Subsystem Documentation

@src/attractor/agent/CLAUDE.md — Autonomous coding agent: session lifecycle, agentic loop, tool execution, and provider profiles.
@src/attractor/llm/CLAUDE.md — Provider-agnostic LLM client: adapters for Anthropic/OpenAI/Gemini, middleware, streaming.
@src/attractor/pipeline/CLAUDE.md — DAG-based pipeline engine: DOT parsing, node handlers, condition routing, checkpointing.
@tests/CLAUDE.md — Test suite conventions: structure, fixtures, async patterns, coverage requirements.

## Quick Reference

```bash
# Setup
bash .claude/scripts/install.sh          # Full install with verification
source .venv/bin/activate                 # Activate venv

# Run
attractor run pipeline.dot --model gpt-4o --verbose
attractor validate pipeline.dot --strict
attractor resume checkpoint.json --pipeline-dot pipeline.dot

# Dev
pytest -v --cov=src/attractor             # Tests with coverage
black . --line-length 88                  # Format
ruff check .                              # Lint
mypy . --strict                           # Type check
```

## Architecture Overview

```
CLI (click)
 └─ pipeline/ ── Parses DOT → validates → executes node graph
      └─ agent/ ── CodergenHandler invokes Session for LLM-driven coding
           └─ llm/ ── Routes requests to provider adapters (Anthropic, OpenAI, Gemini)
```

The pipeline engine walks a DOT-defined DAG. Each node dispatches to a handler (e.g., `CodergenHandler`). Handlers can invoke `agent.Session`, which runs an agentic loop: prompt the LLM, execute tool calls, repeat until the model produces a text-only response. The LLM client routes to the correct provider adapter based on model name.

## Project Conventions

- **Python >= 3.11** required. Uses `|` union syntax, `match/case`, `from __future__ import annotations`.
- **Strict mypy**. All code fully typed. Use `Protocol` for structural typing.
- **Async-first**. All I/O is async. Use `asyncio` for concurrency.
- **Dataclasses** for data structures. No Pydantic models in core (Pydantic is a dependency for validation utilities only).
- **Protocol-based extensibility**. New handlers, adapters, and environments implement protocols — no base-class inheritance required.
- **PEP 8 + Black** (88 char line length). Import order: stdlib → third-party → local.
- **Google-style docstrings** with Args/Returns/Raises sections.

## Key Entry Points

- **CLI**: `src/attractor/cli.py` → Click group with `run`, `validate`, `resume` subcommands
- **Package entry point**: `attractor = "attractor.cli:main"` (defined in `pyproject.toml`)
- **Agent API**: `Session.submit(prompt)` → `AsyncIterator[AgentEvent]`
- **LLM API**: `LLMClient.complete(request)` / `.generate(prompt, model)` / `.stream(request)`
- **Pipeline API**: `PipelineEngine.run(pipeline)` → `PipelineContext`

## Dependencies

| Package | Purpose |
|---|---|
| `click` | CLI framework |
| `rich` | Terminal formatting (tables, colors, logging) |
| `httpx` | HTTP client |
| `anthropic` | Claude API adapter |
| `openai` | OpenAI API adapter |
| `google-genai` | Gemini API adapter |
| `pydot` | GraphViz DOT file parsing |
| `pydantic` | Data validation utilities |
| `anyio` | Async primitives |

## Environment Variables

API keys are loaded from the environment (or `.env` file):
- `ANTHROPIC_API_KEY` — Required for Claude models
- `OPENAI_API_KEY` — Required for GPT models
- `GOOGLE_API_KEY` — Required for Gemini models

## Common Gotchas

- **Adapter lazy-loading**: `LLMClient._default_adapters()` silently skips providers whose SDK isn't installed. If a model fails to route, check that the provider package is installed.
- **Anthropic message alternation**: The Anthropic adapter inserts synthetic placeholder messages to enforce the strict user/assistant alternation the API requires.
- **Start node detection**: Pipeline parser auto-detects start nodes by `start=true` attribute or node named `"start"`. If neither exists, validation fails.
- **CodergenHandler fallback**: If `agent.Session` can't be imported, the handler falls back to an echo stub — useful for pipeline-only testing.
- **Checkpoint naming**: Files use `checkpoint_{timestamp_ms}.json`. The `resume` command finds the latest automatically.
