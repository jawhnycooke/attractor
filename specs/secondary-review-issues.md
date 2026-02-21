# Secondary Spec Compliance Review — Consolidated Issues

## Critical / High Priority (Actionable Now)

### Condition Expression Language

| ID | Sev | Summary | File |
|---|---|---|---|
| C1 | CRIT | Condition syntax uses Python `==`/`and` instead of spec `=`/`&&` | conditions.py |
| C2 | CRIT | `outcome` and `preferred_label` not first-class condition variables | conditions.py |
| C3 | HIGH | `or` and `not` operators accepted but spec explicitly excludes them | conditions.py |
| C4 | MED | Missing `context.` prefix fallback resolution | conditions.py |

### Engine Edge Selection

| ID | Sev | Summary | File |
|---|---|---|---|
| E1 | HIGH | Step 1 does not short-circuit on condition match — mixes conditional+unconditional | engine.py:327-352 |
| E2 | MED | Label normalization missing accelerator prefix stripping | engine.py:356-359 |
| E15 | MED | Steps 4/5 operate on mixed set instead of unconditional-only | engine.py:371-379 |
| E16 | LOW | Final fallback should try any edge, not raise error | engine.py:346-352 |

### Engine Retry / Failure Routing

| ID | Sev | Summary | File |
|---|---|---|---|
| E4 | HIGH | Node max_retries=0 ignores graph default_max_retry (should inherit) | engine.py:136 |
| E5 | HIGH | Retry loop: no SUCCESS reset, no allow_partial, no exception catch | engine.py:139-155 |
| E6 | MED | Backoff max_delay 30s not 60s; jitter range ±25% not ±50% | engine.py:383-392 |
| E7 | HIGH | Unrouted FAIL silently falls through to normal edge selection | engine.py:212-222 |

### Engine Goal Gates

| ID | Sev | Summary | File |
|---|---|---|---|
| E3 | HIGH | Terminal node handler executed before goal gate check (spec checks first) | engine.py:115 |
| E9 | HIGH | Goal gate checks completion, not outcome status (FAIL passes gate) | engine.py:234-241 |
| E10 | HIGH | Goal gate retries skip node-level retry_target, go straight to pipeline-level | engine.py:250-262 |
| E11 | HIGH | Unsatisfied goal gate with no retry target silently completes (should error) | engine.py:262-275 |

### Engine Misc

| ID | Sev | Summary | File |
|---|---|---|---|
| E12 | MED | loop_restart edge attribute never checked | engine.py |
| E13 | MED | FAIL status with no outgoing fail edge does not raise error | engine.py |
| E14 | LOW | SKIPPED status not handled (should skip outcome recording) | engine.py |

### Handler Interface

| ID | Sev | Summary | File |
|---|---|---|---|
| H1 | HIGH | Handler signature is (node, context) not spec (node, context, graph, logs_root) | handlers.py:30-36 |
| H2 | HIGH | CodergenHandler writes no files (prompt.md, response.md, status.json) | handlers.py:112-191 |
| H3 | MED | CodergenHandler does not fall back to node.label when prompt empty | handlers.py:127 |
| H4 | MED | CodergenHandler does not expand $goal variable | handlers.py:131-132 |
| H5 | MED | CodergenHandler sets `last_codergen_output` not `last_response` | handlers.py:176 |
| H6 | HIGH | WaitHumanHandler uses preferred_label not suggested_next_ids; wrong context keys | handlers.py:237-241 |
| H7 | MED | WaitHumanHandler no accelerator key parsing from edge labels | handlers.py:234 |
| H8 | MED | WaitHumanHandler no timeout/skip handling | handlers.py:229-248 |
| H10 | LOW | ConditionalHandler evaluates edges (should be pure no-op) | handlers.py:266-294 |
| H11 | MED | ParallelHandler uses branches attr not outgoing edges; no join/error policies | handlers.py:313-383 |
| H12 | MED | FanInHandler trivial first-found heuristic; no outcome ranking | handlers.py:386-409 |
| H13 | MED | ToolHandler falls back to `command` attr (spec only uses `tool_command`) | handlers.py:427 |
| H14 | MED | ManagerLoopHandler simplified; missing full supervisor pattern | handlers.py:473-530 |
| H15 | LOW | HandlerRegistry no default_handler fallback | handlers.py:39-83 |

### Validator

| ID | Sev | Summary | File |
|---|---|---|---|
| V1 | HIGH | Missing stylesheet_syntax validation rule | validator.py |
| V2 | HIGH | Missing fidelity_valid validation rule | validator.py |
| V3 | HIGH | reachability rule uses WARNING; spec requires ERROR | validator.py:271-279 |
| V4 | HIGH | type_known rule uses ERROR; spec requires WARNING | validator.py:181-191 |
| V5 | HIGH | No check for exactly one start node (multiple accepted) | validator.py + parser.py |
| V6 | MED | No check for exactly one exit node | validator.py |
| V7 | MED | Missing validate_or_raise() function | validator.py |
| V8 | LOW | LintRule protocol missing name attribute | validator.py:63-67 |

### Parser

| ID | Sev | Summary | File |
|---|---|---|---|
| P1 | HIGH | Parser does not reject undirected graphs or `strict` modifier | parser.py:59-73 |
| P2 | HIGH | Parser does not reject multiple graphs in single file | parser.py:69-73 |
| P3 | MED | Chained edges rely on pydot without explicit test coverage | parser.py |
| P4 | MED | Graph-level `label` not extracted as Pipeline field | models.py + parser.py |
| P5 | MED | Start node detection misses "Start" (capitalized) | parser.py:357-376 |
| P6 | MED | Terminal detection misses nodes named "exit" or "end" | parser.py:379-391 |
| P7 | MED | outgoing_edges sorts by weight only, no lexical tiebreak | models.py:166-172 |
| X38 | LOW | CodergenHandler reads from attributes dict, not node.prompt field | handlers.py:127 |

### Stylesheet

| ID | Sev | Summary | File |
|---|---|---|---|
| S3 | MED | Stylesheet resolution ignores graph-level default attributes | stylesheet.py |
| S4 | MED | Stylesheet not applied as transform between parsing and validation | stylesheet.py |
| S5 | MED | apply_stylesheet doesn't set PipelineNode fields directly | stylesheet.py:226-247 |

### Context / State

| ID | Sev | Summary | File |
|---|---|---|---|
| S1-ctx | MED | PipelineContext not thread-safe; missing get_string, clone, snapshot, append_log | models.py:210-314 |
| S3-ckpt | LOW | node_retries checkpoint field never populated by engine | engine.py:394-411 |
| S4-ckpt | LOW | logs checkpoint field never populated | models.py:336 |

## Not Implemented (Documented in specs/)

These were already identified and documented in specs/. Confirmed still missing:

| Area | Gap Docs | Status |
|---|---|---|
| Interviewer protocol (I1-I3) | specs/interviewer-protocol.md | Not implemented |
| Status file contract | specs/status-file-contract.md | Not implemented |
| Run directory structure | specs/run-directory.md | Not implemented |
| Context fidelity modes | specs/context-fidelity.md | Not implemented |
| AST transform pipeline | — | Not implemented |
| Event system (§9.6) | — | Not implemented |
| Artifact store (§5.5) | — | Not implemented |
| HTTP server mode (§9.5) | — | Not implemented |
| Tool call hooks (§9.7) | — | Not implemented |

## LLM/Agent Subsystem Issues

| ID | Sev | Summary | File |
|---|---|---|---|
| L2 | HIGH | Request missing provider, tool_choice, provider_options fields | llm/models.py |
| L8 | HIGH | Anthropic adapter lacks auto cache_control injection | anthropic_adapter.py |
| L12 | HIGH | No error hierarchy — retries all errors indiscriminately | llm/client.py |
| L13 | HIGH | Retry policy retries non-retryable errors (401, 403) | llm/client.py |
| A2 | HIGH | Tool calls in agent loop executed sequentially not concurrently | agent/loop.py |
| A5 | MED | max_turns conflated with max_tool_rounds_per_input | agent/session.py |
| A20 | HIGH | Subagent tools are stubs, not functional | agent/tools/subagent.py |
| L1 | MED | Message missing name and tool_call_id fields | llm/models.py |
| L3 | MED | Response missing provider, raw, warnings, rate_limit fields | llm/models.py |
| L6 | MED | Missing stream event types (TEXT_START, TEXT_END, etc.) | llm/models.py |
| A18 | MED | Tool argument validation against JSON schema not implemented | agent/tools/registry.py |
| A22 | MED | No context window usage tracking or warning | agent/loop.py |
