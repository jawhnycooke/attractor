# CLAUDE.md — Agent Subsystem

The agent subsystem implements the autonomous coding agent. It manages the agentic loop (prompt → tool calls → repeat), tool execution, event streaming, and provider-specific profiles.

## Module Map

| Module | Purpose |
|---|---|
| `session.py` | Top-level API. `Session.submit(prompt)` yields `AgentEvent`s via async iterator. |
| `loop.py` | Core agentic loop. Builds LLM requests, dispatches tool calls, enforces turn limits. |
| `environment.py` | `ExecutionEnvironment` protocol + `LocalExecutionEnvironment` for OS-level file/shell ops. |
| `events.py` | `AgentEvent` types and `EventEmitter` (queue-based async delivery). |
| `turns.py` | Turn types: `UserTurn`, `AssistantTurn`, `ToolTurn` for conversation history modeling. |
| `loop_detection.py` | `LoopDetector` — fingerprints tool calls, detects repeating patterns (length 1–3). |
| `truncation.py` | Two-stage output truncation (chars first, then lines). Keeps head + tail. |
| `prompts.py` | System prompt construction with context interpolation. |
| `profiles/` | Provider-specific tool sets and system prompt templates (`AnthropicProfile`, `OpenAIProfile`, `GeminiProfile`). |
| `tools/` | Tool implementations: `core_tools.py` (read/write/edit/shell/grep/glob), `apply_patch.py`, `subagent.py`, `registry.py`. |

## Key Patterns

- **Session lifecycle**: `IDLE → PROCESSING → AWAITING_INPUT` (or `CLOSED`). Submit queues a background `asyncio.Task` and yields events as they arrive.
- **Tool registry**: `Session._build_registry()` wires tools based on the active `ProviderProfile.tool_definitions`. Not all providers expose all tools.
- **Loop termination**: The loop exits on (1) text-only response or (2) turn/tool-round limit. Loop detection emits a warning event but does **not** terminate the loop.
- **Loop detection window**: Checks last N tool call fingerprints (name + args hash). When a repeating pattern is detected, a `LOOP_DETECTION` event is emitted as a warning.
- **Truncation order**: Stage 1 (char limit) applied first, Stage 2 (line limit) applied to the result. Middle content omitted with `...` marker. Each tool has independent limits.
- **Concurrent tool execution**: When the LLM returns multiple tool calls in one response, they execute concurrently via `asyncio.gather`.
- **Subagent tools**: `spawn_agent`, `send_input`, `wait`, `close_agent` enable multi-agent coordination. Depth is capped by `SessionConfig.max_subagent_depth`.

## ExecutionEnvironment Protocol

All file and shell operations go through `ExecutionEnvironment`. This abstraction enables testing with mock environments and future remote execution.

Methods: `initialize`, `read_file`, `write_file`, `file_exists`, `list_directory`, `exec_command`, `grep`, `glob`, `cleanup`.

Note: `apply_patch` is a standalone tool in `tools/apply_patch.py`, not an `ExecutionEnvironment` method.

`LocalExecutionEnvironment` runs commands with signal handling and filters sensitive env vars from subprocess environments.

## Adding a New Tool

1. Define the tool handler function in `tools/core_tools.py` (or a new file).
2. Create a `ToolDefinition` with name, description, and JSON schema for parameters.
3. Add the `(handler, definition)` pair to `Session._build_registry._tool_map`.
4. Add the tool name to the relevant `ProviderProfile.tool_definitions`.
5. Add truncation limits in `truncation.py` if the tool can produce large output.
