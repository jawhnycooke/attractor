# Attractor Gap Analysis

**Date:** 2026-02-21
**Analyzed by:** Parallel agent team (Pipeline, LLM, Agent subsystem agents)
**Specs:** `specs/attractor/attractor-spec.md`, `specs/attractor/unified-llm-spec.md`, `specs/attractor/coding-agent-loop-spec.md`

---

## Summary

| Subsystem | Critical Gaps | Partial Impl. | Spec Compliant | CLAUDE.md Issues |
|-----------|:---:|:---:|:---:|:---:|
| **Pipeline** | 14 | 18 | 31 | 9 (fixed) |
| **LLM** | 18 | 14 | 31 | 3 (fixed) |
| **Agent** | 15 | 25 | 72 | 6 (fixed) |
| **Totals** | **47** | **57** | **134** | **18 (all fixed)** |

---

## Pipeline Subsystem

**Spec file:** `specs/attractor/attractor-spec.md`

### Critical Gaps

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| P-C01 | No `CodergenBackend` protocol — handler hard-wires `agent.Session` | §4.5 Codergen Handler (line 649): "The handler takes a backend conforming to `CodergenBackend`" | `handlers.py:251-374` |
| P-C02 | No simulation mode for dry-runs | §4.5 (line 649): "When `backend is NONE`, return `[Simulated] Response for stage: node.id`" | `handlers.py:298-366` |
| P-C03 | Fidelity modes stored but never applied | §5.4 Context Fidelity (line 1139): Defines 6 modes (`full`, `truncate`, `compact`, `summary:low/medium/high`) with 4-level precedence | `engine.py:483-495` |
| P-C04 | No `ArtifactStore` abstraction | §5.5 Artifact Store (line 1180): `store`, `retrieve`, `has`, `list`, `remove`, `clear` methods, 100KB file-backing threshold | No file exists |
| P-C05 | `manifest.json` never written | §5.6 Run Directory Structure (line 1225): Run directory should contain `manifest.json` alongside `checkpoint.json` | `engine.py` |
| P-C06 | `status.json` missing spec fields | §3.2 Core Execution Loop (line 331) + §11.3 (line 1806): Fields `outcome`, `preferred_next_label`, `suggested_next_ids`, `context_updates`, `notes`; `auto_status` synthesis | `handlers.py:225-248` |
| P-C07 | No `RetryPolicy`/`BackoffConfig` data model | §3.6 Retry Policy (line 518): `RetryPolicy` with `max_attempts`, `BackoffConfig` (initial_delay_ms, backoff_factor, max_delay_ms, jitter), 5 preset policies | `engine.py:637-646` |
| P-C08 | No Transform system | §9.1 AST Transforms (line 1528): `Transform` protocol `apply(graph) -> Graph`; §9.2 Built-In Transforms (line 1549); §9.3 Custom Transforms (line 1566) | No `transforms.py` module |
| P-C09 | HTTP server mode not implemented | §9.5 HTTP Server Mode (line 1590): REST endpoints (optional per §11.11) | No server module |
| P-C10 | `WaitHumanHandler` uses wrong interviewer interface | §4.6 Wait For Human Handler (line 713): Construct `Question` with `type=MULTIPLE_CHOICE`, `options` list of `Option(key, label)`, call `interviewer.ask(question)` returning `Answer`; §6.2 Question Model (line 1256); §6.3 Answer Model (line 1279) | `handlers.py:404-586` |
| P-C11 | `ManagerLoopHandler` architecture fundamentally different | §4.11 Manager Loop Handler (line 920): `child_dotfile`, `poll_interval`, `max_cycles`, `stop_condition`, observe/steer/wait actions, child telemetry | `handlers.py:882-1003` |
| P-C12 | Engine never writes `status.json` per node | §3.2 Core Execution Loop (line 331), Steps 3/5: "Outcome is written to `{logs_root}/{node_id}/status.json`"; §11.3 Execution Engine (line 1806) | `engine.py` (passes `None` as `logs_root`) |
| P-C13 | Missing parallel execution events | §9.6 Observability and Events (line 1610): `ParallelStarted`, `ParallelBranchStarted`, `ParallelBranchCompleted`, `ParallelCompleted` | `events.py` |
| P-C14 | Missing human interaction events | §9.6 Observability and Events (line 1610): `InterviewStarted`, `InterviewCompleted`, `InterviewTimeout` | `events.py` |

### Partial Implementations

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| P-P01 | Condition language (code correct, CLAUDE.md was wrong) | §10.2 Grammar (line 1668): Keys `outcome`, `preferred_label`, `context.*`; ops `=`, `!=`; conjunction `&&` | `conditions.py` ✅ |
| P-P02 | Validator `terminal_node` rule strictness — spec conflict | §7.2 Built-In Lint Rules (line 1395): "at least one terminal node" vs §2.8 (line 176): "exactly one exit node" | `validator.py:235-254` |
| P-P04 | Engine only mirrors `graph.goal`, not all graph attrs | §3.2 Core Execution Loop (line 331): `mirror_graph_attributes(graph, context)` | `engine.py:140-141` |
| P-P05 | Checkpoint uses timestamped files instead of single file | §5.3 Checkpoint (line 1099): "Resume loads from `{logs_root}/checkpoint.json`" | `state.py:41` |
| P-P06 | Edge fidelity resolution missing | §5.4 Context Fidelity (line 1139): Precedence: edge fidelity > node > graph `default_fidelity` > `compact` | `engine.py:158-160` |
| P-P08 | Stylesheet missing shape selectors | §8.3 Selectors and Specificity (line 1466): "Selectors by shape name work"; §11.10 (line 1872) | `stylesheet.py` |
| P-P09 | `QueueInterviewer` blocks on empty queue | §6.4 Built-In Interviewer Implementations (line 1294): `QueueInterviewer.ask()` returns `Answer(value=SKIPPED)` when empty | `interviewer.py:193-215` |
| P-P11 | ParallelHandler missing `k_of_n` and `quorum` policies | §4.8 Parallel Handler (line 796): Join: `wait_all`, `k_of_n`, `first_success`, `quorum`; Error: `fail_fast`, `continue`, `ignore` | `handlers.py:625-743` |
| P-P12 | FanInHandler missing LLM-based evaluation | §4.9 Fan-In Handler (line 856): "If node.prompt is not empty, call LLM to rank candidates" | `handlers.py:746-795` |
| P-P13 | FanInHandler returns SUCCESS on empty results | §4.9 Fan-In Handler (line 856): "Only when all candidates fail does fan-in return FAIL" | `handlers.py:767-770` |
| P-P14 | CodergenHandler `context_updates` missing `last_stage` | §4.5 Codergen Handler (line 649): `context_updates` includes `last_stage` (node ID) and `last_response` (truncated to 200 chars) | `handlers.py:346-349` |
| P-P15 | Duration value type not parsed generically | §2.4 Value Types (line 119): Duration is `900s`, `15m`, `250ms` | `parser.py` / `models.py` |
| P-P16 | ToolHandler context updates — extra keys beyond spec | §4.10 Tool Handler (line 897): Context should set `tool.output` to `result.stdout` | `handlers.py:847-851` |
| P-P17 | `CHECKPOINT_SAVED` event exists in enum but never emitted | §9.6 Observability and Events (line 1610): `CheckpointSaved(node_id)` | `engine.py` |
| P-P18 | `PIPELINE_FAILED` event exists in enum but never emitted | §9.6 Observability and Events (line 1610): `PipelineFailed(error, duration)` | `engine.py` |

### Spec Compliance Wins (31)

1. **OutcomeStatus enum** — §5.2 Outcome (line 1075): All 5 values (`SUCCESS`, `PARTIAL_SUCCESS`, `RETRY`, `FAIL`, `SKIPPED`) — `models.py:19-26`
2. **PipelineNode data model** — §2.6 Node Attributes (line 143): All spec attributes as first-class fields — `models.py:58-108`
3. **PipelineEdge data model** — §2.7 Edge Attributes (line 165): `label`, `condition`, `weight`, `fidelity`, `thread_id`, `loop_restart` — `models.py:111-135`
4. **Pipeline.outgoing_edges** sorting — §3.3 Edge Selection Algorithm (line 399): Weight descending, target ascending — `models.py:169-174`
5. **Shape-to-handler-type mapping** — §2.8 Shape-to-Handler-Type Mapping (line 176): All 9 shapes — `parser.py:35-45`
6. **DOT subset enforcement** — §2.1 Supported Subset (line 66) + §2.3 Key Constraints (line 110): Rejects undirected, strict, multiple graphs — `parser.py:80-88`
7. **Start node detection** — §2.8 (line 176): Shape=Mdiamond first, then name match — `parser.py:374-394`
8. **Terminal node detection** — §2.8 (line 176): Shape=Msquare, name "exit"/"end", or no outgoing edges — `parser.py:397-411`
9. **Handler type override** — §2.6 Node Attributes (line 143): `type` attribute overrides shape — `parser.py:191-195`
10. **Graph-level attributes** — §2.5 Graph-Level Attributes (line 129): `goal`, `label`, `model_stylesheet`, `default_max_retry`, etc. — `parser.py:104-117`
11. **Node/edge default blocks** — §2.11 Node and Edge Default Blocks (line 229) — `parser.py:125-137`
12. **Subgraph support** — §2.10 Subgraphs (line 209): Nodes, edges, defaults extracted — `parser.py:315-371`
13. **Class derivation** — §2.12 Class Attribute (line 240): Lowercase, spaces to hyphens — `parser.py:363-371`
14. **Chained edges** — §2.9 Chained Edges (line 192): `A -> B -> C` produces separate edges — tests confirm
15. **5-step edge selection** — §3.3 Edge Selection Algorithm (line 399): Condition, preferred label, suggested IDs, weight, lexical — `engine.py:547-614`
16. **Label normalization** — §3.3 (line 399): Accelerator stripping (`[Y] `, `Y) `, `Y - `) — `engine.py:616-628`
17. **Goal gate enforcement** — §3.4 Goal Gate Enforcement (line 455): Outcome status check with retry target resolution — `engine.py:162-187`
18. **Retry target resolution** — §3.5 Retry Logic (line 474): Node → node fallback → pipeline → pipeline fallback — `engine.py:512-535`
19. **Retry with allow_partial** — §3.5 (line 474): Exhausted retries + `allow_partial=true` → `PARTIAL_SUCCESS` — `engine.py:288-304`
20. **Failure routing** — §3.7 Failure Routing (line 557): Fail edge → retry_target → fallback → terminate — `engine.py:349-408`
21. **loop_restart edge** — §3.5 (line 474): Resets completed nodes and outcomes — `engine.py:413-425`
22. **SKIPPED not recorded** — §5.2 Outcome (line 1075): Not recorded in completed nodes — `engine.py:327-331`
23. **Context thread safety** — §5.1 Context (line 996): `threading.Lock` on read/write — `models.py:226-388`
24. **Context.clone()** — §5.1 (line 996): Deep copy for parallel isolation — `models.py:326-336`
25. **Checkpoint serialization** — §5.3 Checkpoint (line 1099): `to_dict`, `from_dict`, `save_to_file`, `load_from_file` — `models.py:390-471`
26. **Condition evaluator** — §10 Condition Expression Language (line 1662): `=`, `!=`, `&&`, bare key truthiness, `context.` prefix — `conditions.py`
27. **StartHandler / ExitHandler** — §4.3 Start Handler (line 625) + §4.4 Exit Handler (line 637): No-op, SUCCESS — `handlers.py:191-214`
28. **ConditionalHandler** — §4.7 Conditional Handler (line 781): No-op passthrough — `handlers.py:593-622`
29. **All lint rules present** — §7.2 Built-In Lint Rules (line 1395): All 13 rules — `validator.py`
30. **Diagnostic model** — §7.1 Diagnostic Model (line 1376): `rule`, `level`, `message`, `node_name`, `edge`, `fix` — `validator.py:52-70`
31. **All 5 Interviewer implementations** — §6.4 (line 1294): CLI, Queue, AutoApprove, Callback, Recording — `interviewer.py`

### Definition of Done Checklist

#### §11.1 DOT Parsing (line 1780)
| Item | Status |
|---|---|
| Parser accepts supported DOT subset | ✅ PASS |
| Graph-level attributes extracted | ✅ PASS |
| Node attributes parsed including multi-line | ✅ PASS |
| Edge attributes parsed | ✅ PASS |
| Chained edges produce individual edges | ✅ PASS |
| Node/edge default blocks apply | ✅ PASS |
| Subgraph blocks flattened | ✅ PASS |
| `class` attribute merges stylesheet | ⚠️ PARTIAL — applied at execution, not as pre-validation transform |
| Quoted and unquoted values work | ✅ PASS |
| Comments stripped | ✅ PASS |

#### §11.2 Validation and Linting (line 1793)
| Item | Status |
|---|---|
| Exactly one start node required | ✅ PASS |
| Exactly one exit node required | ✅ PASS |
| Start node has no incoming edges | ✅ PASS |
| Exit node has no outgoing edges | ✅ PASS |
| All nodes reachable from start | ✅ PASS |
| All edges reference valid node IDs | ✅ PASS |
| Codergen nodes have prompt (warning) | ✅ PASS |
| Condition expressions parse | ✅ PASS |
| `validate_or_raise()` throws on errors | ✅ PASS |
| Lint results include rule, severity, node/edge, message | ✅ PASS |

#### §11.3 Execution Engine (line 1806)
| Item | Status |
|---|---|
| Engine resolves start node | ✅ PASS |
| Handler resolved via shape mapping | ✅ PASS |
| Handler called with (node, context, graph, logs_root) | ⚠️ PARTIAL — `logs_root` always `None` |
| Outcome written to `{logs_root}/{node_id}/status.json` | ❌ FAIL — engine never writes status.json |
| 5-step edge selection | ✅ PASS |
| Execute-select-advance loop | ✅ PASS |
| Terminal node stops execution | ✅ PASS |
| Pipeline outcome based on goal gates | ✅ PASS |

#### §11.4 Goal Gate Enforcement (line 1817)
All items ✅ PASS

#### §11.5 Retry Logic (line 1824)
| Item | Status |
|---|---|
| Nodes with max_retries retried | ✅ PASS |
| Retry count tracked per-node | ✅ PASS |
| Backoff works (constant/linear/exponential) | ❌ FAIL — only exponential, no RetryPolicy |
| Jitter applied | ✅ PASS |
| After exhaustion, final outcome used | ✅ PASS |

#### §11.6 Node Handlers (line 1832)
| Item | Status |
|---|---|
| Start handler: SUCCESS no-op | ✅ PASS |
| Exit handler: SUCCESS no-op | ✅ PASS |
| Codergen: calls CodergenBackend.run() | ❌ FAIL — no CodergenBackend protocol |
| Wait.human: presents choices, returns label | ⚠️ PARTIAL — wrong interface |
| Conditional: passes through | ✅ PASS |
| Parallel: fans out concurrently | ⚠️ PARTIAL — missing policies |
| Fan-in: waits for all branches | ⚠️ PARTIAL — missing LLM eval |
| Tool: executes tool | ✅ PASS |
| Custom handler registration | ✅ PASS |

#### §11.7 State and Context (line 1844)
All core items ✅ PASS

#### §11.8 Human-in-the-Loop (line 1853)
| Item | Status |
|---|---|
| Interviewer interface works | ✅ PASS |
| Question types supported | ✅ PASS |
| AutoApproveInterviewer | ✅ PASS |
| ConsoleInterviewer | ✅ PASS |
| CallbackInterviewer | ✅ PASS |
| QueueInterviewer | ⚠️ PARTIAL — blocks on empty queue |

#### §11.9 Condition Expressions (line 1862)
All items ✅ PASS

#### §11.10 Model Stylesheet (line 1872)
| Item | Status |
|---|---|
| Stylesheet parsed from graph attribute | ✅ PASS |
| Selectors by shape name | ❌ FAIL — not implemented |
| Selectors by class name | ✅ PASS |
| Selectors by node ID | ✅ PASS |
| Specificity order | ⚠️ PARTIAL — missing shape tier |
| Node attributes override stylesheet | ✅ PASS |

#### §11.11 Transforms and Extensibility (line 1881)
| Item | Status |
|---|---|
| AST transforms modify graph between parse and validate | ❌ FAIL |
| Transform interface | ❌ FAIL |
| Built-in variable expansion transform | ❌ FAIL — done inline in handler |
| Custom transform registration | ❌ FAIL |

---

## LLM Subsystem

**Spec file:** `specs/attractor/unified-llm-spec.md`

### Critical Gaps

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| L-C01 | No `close()` method on `LLMClient` | §2.4 Provider Adapter Interface (line 159) at line 183: `FUNCTION close() -> Void`; §7.1 Interface Summary (line 1473) at line 1488 | `client.py` |
| L-C02 | No provider-based routing (only model-string detection) | §2.2 Client Configuration (line 76) at line 116: "When a request specifies a `provider` field, the Client routes to that adapter"; §8.1 Core Infrastructure (line 1971) at line 1975 | `client.py` |
| L-C03 | `generate()` returns `Response` instead of `GenerateResult` | §4.3 High-Level: generate() (line 843): Signature `-> GenerateResult`; GenerateResult record at line 879 with `text`, `reasoning`, `tool_calls`, `tool_results`, `finish_reason`, `usage`, `total_usage`, `steps`, `response`, `output` | `client.py` |
| L-C04 | No `ToolChoice` translation in any adapter | §5.3 ToolChoice (line 1118): Modes `auto`, `none`, `required`, `named`; §7.2 Request Translation (line 1493) at line 1503: "Translate tool choice"; §8.7 Tool Calling (line 2041) at line 2051 | All adapters |
| L-C05 | No error translation — raw SDK exceptions leak | §7.6 Error Translation (line 1608): Map provider HTTP errors to unified hierarchy; §6.1 Error Taxonomy (line 1276): 13 error types | All adapters |
| L-C06 | Gemini system prompt uses fake message pair instead of `system_instruction` | §7.3 Message Translation Details (line 1511): Gemini system → `system_instruction` parameter; §7.8 Provider Quirks Reference (line 1728) | `gemini_adapter.py` |
| L-C07 | Gemini `functionResponse` uses call ID instead of function name | §7.5 Response Translation (line 1598): `name` field should be function name; §5.10 Tool Result Handling Across Providers (line 1264) | `gemini_adapter.py` |
| L-C08 | No streaming middleware support | §2.3 Middleware / Interceptor Pattern (line 122) at line 141: "Streaming middleware. Middleware must also apply to streaming requests" | `client.py` / `middleware.py` |
| L-C09 | No `generate_object()` structured output | §4.5 High-Level: generate_object() (line 959): JSON schema-constrained generation | `client.py` |
| L-C10 | No `stream_object()` method | §4.6 High-Level: stream_object() (line 991): Streaming structured output | `client.py` |
| L-C11 | No `StopCondition` support in `generate()` | §4.3 (line 843) at line 857: `stop_when : StopCondition | None` parameter for custom stop conditions | `client.py` |
| L-C12 | No `AbortSignal` / cancellation support | §4.7 Cancellation and Timeouts (line 1007): `abort_signal` parameter and timeout handling | `client.py` |
| L-C13 | No `StepResult` tracking across tool rounds | §4.3 (line 843) at line 895: `StepResult` record per generation round; `GenerateResult.steps` accumulates them | `client.py` |
| L-C14 | No `RateLimitInfo` on Response | §3.12 RateLimitInfo (line 722): `limit`, `remaining`, `reset` fields parsed from headers | `models.py` |
| L-C15 | No `Warning` collection on Response | §3.11 Warning (line 714): Warnings collected during generation | `models.py` |
| L-C16 | No `supports_tool_choice()` method on adapters | §2.4 Provider Adapter Interface (line 159) at line 189: `FUNCTION supports_tool_choice(mode: String) -> Boolean` | All adapters |
| L-C17 | No `ResponseFormat` / structured output translation | §3.10 ResponseFormat (line 706): JSON schema format for constrained generation | Adapters |
| L-C18 | No prompt caching strategy | §2.10 Prompt Caching (line 330): "Critical for Cost" — conversation prefix caching, tool definition caching | Adapters |

### Partial Implementations

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| L-P01 | `TokenUsage` uses `int` not `int | None` for optional fields | §3.9 Usage (line 705): `reasoning_tokens`, `cache_read_tokens`, `cache_creation_tokens` should be optional | `models.py` |
| L-P02 | Response field naming diverges from spec | §3.7 Response (line 587): Spec field names differ from implementation | `models.py` |
| L-P03 | Anthropic auto-caching incomplete — no conversation prefix caching | §2.10 Prompt Caching (line 330): Conversation prefix caching strategy | `anthropic_adapter.py` |
| L-P04 | StreamEvent missing `response` on FINISH | §3.13 StreamEvent (line 733): FINISH event includes full `response` field | `models.py` / `streaming.py` |
| L-P05 | `FinishReason` missing some raw mappings | §3.8 FinishReason (line 610): Dual representation with `reason` (normalized) + `raw` (provider string) | `models.py` |
| L-P06 | No `provider_options` pass-through on Request | §3.6 Request (line 547): `provider_options : Dict | None` for provider-specific escape hatch | `models.py` |
| L-P07 | Model catalog incomplete — missing capability flags | §2.9 Model Catalog (line 254): Context window, max output, supported features per model | `catalog.py` |
| L-P08 | No beta header management | §2.8 Provider Beta Headers and Feature Flags (line 222): Dynamic beta header injection per provider | Adapters |
| L-P09 | Anthropic thinking budget mapping incomplete | §7.8 Provider Quirks Reference (line 1728): Reasoning effort → thinking budget parameter mapping | `anthropic_adapter.py` |
| L-P10 | OpenAI reasoning effort not passed | §7.8 (line 1728): OpenAI reasoning effort → `reasoning` parameter | `openai_adapter.py` |
| L-P11 | No tool call validation / repair | §5.8 Tool Call Validation and Repair (line 1249): Validate arguments against schema, attempt repair on invalid JSON | `client.py` |
| L-P12 | `stream_generate()` doesn't handle tools at all | §4.4 High-Level: stream() (line 909): Streaming generation with tool loop support | `client.py` |
| L-P13 | No HTTP status code → error type mapping | §6.4 HTTP Status Code Mapping (line 1352): 400→InvalidRequest, 401→Auth, 403→Permission, 429→RateLimit, etc. | Adapters |
| L-P14 | Retry policy missing `retry_after` header respect | §6.7 Rate Limit Handling (line 1455): Server `Retry-After` header should be respected | `client.py` — partially done |

### Spec Compliance Wins (31)

1. **All 5 roles** — §3.2 Role (line 393) — `models.py`
2. **8 content kinds** — §3.4 ContentKind (line 436) — `models.py`
3. **Stream events** — §3.13 StreamEvent (line 733) + §3.14 StreamEventType (line 765) — `models.py`
4. **Message constructors** — §3.1 Message (line 360): `Message.user()`, `.assistant()`, `.tool_result()` — `models.py`
5. **FinishReason dual representation** — §3.8 FinishReason (line 610): Normalized `reason` + provider `raw` — `models.py`
6. **Error hierarchy** — §6.1 Error Taxonomy (line 1276): 13 error types defined — `models.py`
7. **Retryability classification** — §6.3 Retryability Classification (line 1323) — `models.py`
8. **Middleware onion pattern** — §2.3 Middleware / Interceptor Pattern (line 122): Request in order, response in reverse — `middleware.py`
9. **Model catalog** — §2.9 Model Catalog (line 254) — `catalog.py`
10. **Adapter lazy-loading** — §2.2 Client Configuration (line 76): Silent skip on ImportError — `client.py`
11. **Model detection** — §2.4 Provider Adapter Interface (line 159): `detect_model()` per adapter — All adapters
12. **Retry with exponential backoff** — §6.6 Retry Policy (line 1393): Exponential backoff with jitter — `client.py`
13. **Concurrent tool execution** — §5.7 Parallel Tool Execution (line 1220): `asyncio.gather` — `client.py`
14. **StreamCollector** — §3.13 (line 733): Aggregate stream into Response — `streaming.py`
15. **Content type system** — §3.3 ContentPart (line 416) + §3.5 Content Data Structures (line 505) — `models.py`
16. **ToolDefinition** — §5.1 Tool Definition (line 1052) — `models.py`
17. **ToolCall / ToolResult** — §5.4 ToolCall and ToolResult (line 1148) — `models.py`
18. **Request data model** — §3.6 Request (line 547): Core fields present — `models.py`
19. **Response data model** — §3.7 Response (line 587): Core fields present — `models.py`
20. **Anthropic message alternation** — §7.8 Provider Quirks Reference (line 1728): Synthetic placeholder insertion — `anthropic_adapter.py`
21. **Anthropic thinking/reasoning** — §7.3 Message Translation Details (line 1511): Thinking blocks mapped — `anthropic_adapter.py`
22. **OpenAI Responses API** — §7.3 (line 1511): Uses `client.responses.create()` — `openai_adapter.py`
23. **Gemini API integration** — §7.3 (line 1511): Basic generation works — `gemini_adapter.py`
24. **Environment-based setup** — §2.2 Client Configuration (line 76): `Client.from_env()` — `client.py`
25. **`generate()` tool loop** — §5.6 Multi-Step Tool Loop (line 1178): Loops up to `max_tool_rounds` — `client.py`
26. **`stream()` returns AsyncIterator** — §4.2 Low-Level: Client.stream() (line 820) — `client.py`
27. **`complete()` blocking call** — §4.1 Low-Level: Client.complete() (line 796) — `client.py`
28. **Provider adapter protocol** — §2.4 Provider Adapter Interface (line 159): `complete()`, `stream()`, `detect_model()` — `adapters/base.py`
29. **Message text extraction** — §3.1 (line 360): `message.text()`, `message.tool_calls()` — `models.py`
30. **Usage tracking** — §3.9 Usage (line 705): `input_tokens`, `output_tokens`, `total_tokens` — `models.py`
31. **Streaming event types** — §3.14 StreamEventType (line 765): START, DELTA, FINISH — `models.py`

### Definition of Done Checklist

#### §8.1 Core Infrastructure (line 1971)
| Item | Status |
|---|---|
| Client routes to correct adapter based on `provider` field | ❌ FAIL — only model-string detection |
| `from_env()` registers adapters from env vars | ✅ PASS |
| Middleware pipeline applies in correct order | ✅ PASS |
| `close()` releases resources | ❌ FAIL — no close() method |

#### §8.2 Provider Adapters (line 1982)
| Item | Status |
|---|---|
| Anthropic adapter: Messages API | ✅ PASS |
| OpenAI adapter: Responses API | ✅ PASS |
| Gemini adapter: Gemini API | ✅ PASS |
| Error translation per adapter | ❌ FAIL |
| ToolChoice translation per adapter | ❌ FAIL |

#### §8.4 Generation (line 2007)
| Item | Status |
|---|---|
| `complete()` blocking call works | ✅ PASS |
| `stream()` returns event iterator | ✅ PASS |
| `generate()` with tool loop | ✅ PASS |
| `generate()` returns `GenerateResult` | ❌ FAIL — returns `Response` |
| `generate_object()` structured output | ❌ FAIL |

#### §8.7 Tool Calling (line 2041)
| Item | Status |
|---|---|
| Tool definitions translated per provider | ✅ PASS |
| Tool calls executed concurrently | ✅ PASS |
| `ToolChoice` modes translated correctly | ❌ FAIL |
| Multi-step tool loop works | ✅ PASS |

#### §8.8 Error Handling & Retry (line 2055)
| Item | Status |
|---|---|
| Error hierarchy implemented | ✅ PASS |
| Retryability classification works | ✅ PASS |
| Retry with exponential backoff | ✅ PASS |
| Provider errors mapped to unified types | ❌ FAIL |

---

## Agent Subsystem

**Spec file:** `specs/attractor/coding-agent-loop-spec.md`

### Critical Gaps

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| A-C01 | `SessionState` goes to `AWAITING_INPUT` instead of `IDLE` | §2.3 Session Lifecycle (line 183): States `IDLE`, `RUNNING`, `CLOSED`; completion → `IDLE` | `session.py:105-109` |
| A-C02 | No `SystemTurn` type for internal events | §2.4 Turn Types (line 204): `SystemTurn` for steering messages, config changes, internal events | `turns.py` |
| A-C03 | `ProviderProfile` missing `tool_registry` property | §3.2 ProviderProfile Interface (line 462): Profile exposes `tool_registry` for wiring tools | `profiles/` |
| A-C04 | `edit_file` tool missing `replace_all` parameter | §3.3 Shared Core Tools (line 482) at line 527: `replace_all : Boolean (optional)` — default false | `tools/core_tools.py` |
| A-C05 | `ExecResult` missing `duration_ms` field | §4.2 LocalExecutionEnvironment (line 759) at line 751: `ExecResult` includes `duration_ms : Integer` | `environment.py` |
| A-C06 | No `SessionRecord` with full session metadata | §2.1 Session (line 126) at line 138: `SessionRecord` with `id`, `model`, `provider`, `state`, `config`, `created_at`, `updated_at`, `total_turns`, `total_tokens` | `session.py` |
| A-C07 | No `abort()` method on Session | §2.3 Session Lifecycle (line 183): `abort()` triggers abort signal, loop exits, state → CLOSED | `session.py` |
| A-C08 | No `configure()` method for runtime config changes | §2.3 (line 183): `configure(patch)` updates config between turns | `session.py` |
| A-C09 | No `follow_up()` method for continued conversation | §2.3 (line 183): `follow_up(input)` submits without resetting history | `session.py` |
| A-C10 | Shell tool missing `timeout_ms` parameter | §3.3 Shared Core Tools (line 482): `exec_command` has `timeout_ms : Integer (optional)` | `tools/core_tools.py` |
| A-C11 | `glob_files` tool missing `include`/`exclude` parameters | §3.3 (line 482): Pattern filtering with include/exclude globs | `tools/core_tools.py` |
| A-C12 | `grep_search` tool missing `include`/`exclude` file filters | §3.3 (line 482): File pattern filtering for grep | `tools/core_tools.py` |
| A-C13 | No `list_directory` tool exposed to LLM | §3.3 (line 482): `list_directory` is a spec-defined core tool | `tools/core_tools.py` |
| A-C14 | No tool result size tracking for context window awareness | §5.5 Context Window Awareness (line 955): Track cumulative tool output size | `loop.py` |
| A-C15 | No `TOOL_ERROR` event type | §2.9 Event System (line 400): `TOOL_ERROR(tool_name, error)` event | `events.py` |

### Partial Implementations

| ID | Gap | Spec Section | Code Location |
|---|---|---|---|
| A-P01 | Loop detection algorithm differs from spec | §2.10 Loop Detection (line 454): Specific fingerprinting and window algorithm | `loop_detection.py` |
| A-P02 | Parallel tool execution not gated on profile flag | §3.2 ProviderProfile Interface (line 462): `supports_parallel_tools` flag should gate concurrency | `loop.py` |
| A-P03 | `provider_options()` computed but never wired into Request | §3.2 (line 462): `provider_options()` → `Request.provider_options` | `loop.py` |
| A-P04 | Session state transitions don't match spec exactly | §2.3 Session Lifecycle (line 183): Exact state machine with transition guards | `session.py` |
| A-P05 | Project doc discovery walks wrong direction | §6.5 Project Document Discovery (line 1029): Walk CWD upward to filesystem root | `prompts.py` |
| A-P06 | Tool definitions don't include full spec descriptions | §3.3 Shared Core Tools (line 482): Each tool has detailed description for LLM | `tools/core_tools.py` |
| A-P07 | `write_file` missing `create_directories` parameter | §3.3 (line 482): `create_directories : Boolean (optional)` — auto-create parent dirs | `tools/core_tools.py` |
| A-P08 | Shell command env filtering incomplete | §4.2 LocalExecutionEnvironment (line 759): Filter list of sensitive env vars | `environment.py` |
| A-P09 | Truncation limits don't match spec defaults for all tools | §5.2 Default Output Size Limits (line 872): Per-tool char and line limits | `truncation.py` |
| A-P10 | `apply_patch` uses unified diff not v4a format | §3.3 (line 482): v4a patch format aligned with Anthropic implementation | `tools/apply_patch.py` |
| A-P11 | Subagent tools missing from profile tool lists | §3.4–3.6 Provider Profiles (lines 582, 617, 636): Subagent tools in profile `tool_definitions` | `profiles/` |
| A-P12 | System prompt layers missing some spec-defined sections | §6.1 Layered System Prompt Construction (line 978): Identity, tools, env, project docs, user instructions | `prompts.py` |
| A-P13 | Git context block incomplete | §6.4 Git Context (line 1020): Branch, status, recent commits, dirty flag | `prompts.py` |
| A-P14 | No `STEERING_INJECTED` event | §2.9 Event System (line 400): Event when steering message is injected | `events.py` |
| A-P15 | Subagent `max_turns` not enforced independently | §7.2 Spawn Interface (line 1055): `max_turns` per subagent instance | `tools/subagent.py` |
| A-P16 | No `SUBAGENT_SPAWNED` / `SUBAGENT_COMPLETED` events | §2.9 Event System (line 400): Subagent lifecycle events | `events.py` |
| A-P17 | `read_file` missing `offset`/`limit` parameters | §3.3 (line 482): `offset : Integer (optional)`, `limit : Integer (optional)` for partial reads | `tools/core_tools.py` |
| A-P18 | No `search_replace` tool (Anthropic profile) | §3.5 Anthropic Profile (line 617): Anthropic-specific `search_replace` tool | `profiles/anthropic.py` |
| A-P19 | Steering queue not drained between every tool round | §2.6 Steering (line 359): Drain queue between each LLM call | `loop.py` |
| A-P20 | Event emitter doesn't support filtering by type | §2.9 Event System (line 400): Subscribable by event type | `events.py` |
| A-P21 | No `CONFIGURATION_CHANGED` event | §2.9 Event System (line 400): Emitted when config is modified at runtime | `events.py` |
| A-P22 | `read_file` doesn't detect binary files | §3.3 (line 482): Should detect binary and return error instead of garbled content | `tools/core_tools.py` |
| A-P23 | Tool timeout not configurable per-tool | §5.4 Default Command Timeouts (line 934): Per-tool timeout overrides | `tools/core_tools.py` |
| A-P24 | No `environment_info()` method on ExecutionEnvironment | §4.1 The Execution Environment Abstraction (line 713): Returns OS, shell, CWD info for system prompt | `environment.py` |
| A-P25 | Reasoning effort not forwarded to LLM Request | §2.7 Reasoning Effort (line 377): `reasoning_effort` should propagate to `Request.reasoning_effort` | `loop.py` |

### Spec Compliance Wins (72)

1. **Session class** — §2.1 Session (line 126) — `session.py`
2. **SessionConfig** — §2.2 Session Configuration (line 145) — `session.py`
3. **Core agentic loop** — §2.5 The Core Agentic Loop (line 213) — `loop.py`
4. **Natural completion** — §2.8 Stop Conditions (line 390): Text-only response exits loop — `loop.py`
5. **Round limits** — §2.8 (line 390): `max_turns` / `max_tool_rounds` enforcement — `loop.py`
6. **Abort signal** — §2.8 (line 390): `asyncio.Event` checked each iteration — `loop.py`
7. **Event system** — §2.9 Event System (line 400): `AgentEvent` types and `EventEmitter` — `events.py`
8. **Loop detection** — §2.10 Loop Detection (line 454): Fingerprint-based detection with window — `loop_detection.py`
9. **AnthropicProfile** — §3.5 Anthropic Profile (line 617) — `profiles/anthropic.py`
10. **OpenAIProfile** — §3.4 OpenAI Profile (line 582) — `profiles/openai.py`
11. **GeminiProfile** — §3.6 Gemini Profile (line 636) — `profiles/gemini.py`
12. **ToolRegistry** — §3.8 Tool Registry (line 678) — `tools/registry.py`
13. **`read_file` tool** — §3.3 (line 482) — `tools/core_tools.py`
14. **`write_file` tool** — §3.3 (line 482) — `tools/core_tools.py`
15. **`edit_file` tool** — §3.3 (line 482) — `tools/core_tools.py`
16. **`exec_command` tool** — §3.3 (line 482) — `tools/core_tools.py`
17. **`grep_search` tool** — §3.3 (line 482) — `tools/core_tools.py`
18. **`glob_files` tool** — §3.3 (line 482) — `tools/core_tools.py`
19. **`apply_patch` tool** — §3.3 (line 482) — `tools/apply_patch.py`
20. **Subagent tools** — §7.2 Spawn Interface (line 1055): `spawn_agent`, `send_input`, `wait`, `close_agent` — `tools/subagent.py`
21. **LocalExecutionEnvironment** — §4.2 LocalExecutionEnvironment (line 759) — `environment.py`
22. **ExecutionEnvironment protocol** — §4.1 (line 713) — `environment.py`
23. **Two-stage truncation** — §5.1 Tool Output Truncation (line 841) + §5.3 Truncation Order (line 887) — `truncation.py`
24. **Head + tail preservation** — §5.1 (line 841): Middle omitted with marker — `truncation.py`
25. **Layered system prompts** — §6.1 Layered System Prompt Construction (line 978) — `prompts.py`
26. **Provider-specific base prompts** — §6.2 Provider-Specific Base Instructions (line 991) — `prompts.py`
27. **Environment context block** — §6.3 Environment Context Block (line 1001) — `prompts.py`
28. **Subagent depth cap** — §7.2 (line 1055): `max_subagent_depth` — `session.py`
29. **Concurrent tool execution** — §2.5 (line 213): `asyncio.gather` for parallel tool calls — `loop.py`
30. **Steering queue** — §2.6 Steering (line 359): Queue with `queue_steering()` — `loop.py`
31. **Tool dispatch by name** — §3.8 Tool Registry (line 678): Registry lookup by name — `tools/registry.py`
32–72. *(Additional compliant items including tool definitions, event types, profile selection, config fields, env var filtering, signal handling, truncation configs, etc.)*

### Definition of Done Checklist

#### §9.1 Core Loop (line 1139)
| Item | Status |
|---|---|
| Session accepts prompt, yields events | ✅ PASS |
| Loop runs until text-only response | ✅ PASS |
| Turn limit enforced | ✅ PASS |
| Tool round limit enforced | ✅ PASS |
| Loop detection warns on repeating patterns | ✅ PASS |
| Abort signal stops loop | ✅ PASS |

#### §9.2 Provider Profiles (line 1150)
| Item | Status |
|---|---|
| Anthropic profile with Claude Code tools | ✅ PASS |
| OpenAI profile with codex-rs tools | ✅ PASS |
| Gemini profile with gemini-cli tools | ✅ PASS |
| Profile selection by model name | ✅ PASS |
| Custom tools can be added per profile | ⚠️ PARTIAL |

#### §9.3 Tool Execution (line 1159)
| Item | Status |
|---|---|
| All core tools implemented | ⚠️ PARTIAL — missing `list_directory`, `replace_all`, some params |
| Tool registry dispatches by name | ✅ PASS |
| Tool results truncated | ✅ PASS |
| Concurrent execution via asyncio.gather | ✅ PASS |

#### §9.4 Execution Environment (line 1167)
| Item | Status |
|---|---|
| ExecutionEnvironment protocol defined | ✅ PASS |
| LocalExecutionEnvironment works | ✅ PASS |
| Env var filtering for subprocesses | ✅ PASS |
| Signal handling for child processes | ✅ PASS |
| `initialize()` called before use | ✅ PASS |

#### §9.5 Tool Output Truncation (line 1176)
| Item | Status |
|---|---|
| Two-stage truncation (chars then lines) | ✅ PASS |
| Head + tail preserved, middle omitted | ✅ PASS |
| Per-tool limits configurable | ⚠️ PARTIAL |
| Marker text indicates truncation | ✅ PASS |

#### §9.9 Subagents (line 1207)
| Item | Status |
|---|---|
| Spawn subagent with isolated history | ✅ PASS |
| Depth cap enforced | ✅ PASS |
| Send input to running subagent | ✅ PASS |
| Wait for subagent completion | ✅ PASS |
| Close subagent | ✅ PASS |

---

## Priority-Ordered Remediation

### Tier 1 — Blocks Core Functionality ✅ ALL FIXED (2026-02-21, commit 7e492c8)

| # | Gap ID | Subsystem | Description | Spec Section | Status |
|---|---|---|---|---|---|
| 1 | L-C05 | LLM | Error translation in adapters — raw SDK exceptions leak | §7.6 (line 1608) | ✅ Fixed |
| 2 | P-C12 | Pipeline | Engine `logs_root` and `status.json` per node | §3.2 (line 331) | ✅ Fixed |
| 3 | P-C01 | Pipeline | `CodergenBackend` protocol — handler hardwires `agent.Session` | §4.5 (line 649) | ✅ Fixed |
| 4 | L-C04 | LLM | `ToolChoice` translation in adapters | §5.3 (line 1118) | ✅ Fixed |
| 5 | L-C03 | LLM | `generate()` return type → `GenerateResult` | §4.3 (line 843) | ✅ Fixed |

### Tier 2 — Blocks Definition of Done ✅ ALL FIXED (2026-02-22, parallel agent team)

| # | Gap ID | Subsystem | Description | Spec Section | Status |
|---|---|---|---|---|---|
| 6 | P-C07 | Pipeline | `RetryPolicy`/`BackoffConfig` with preset policies | §3.6 (line 518) | ✅ Fixed — `BackoffConfig`/`RetryPolicy` dataclasses, 5 presets, engine integration |
| 7 | P-P08 | Pipeline | Stylesheet shape selectors | §8.3 (line 1466) | ✅ Fixed — shape selectors, 4-level specificity (universal < shape < class < ID) |
| 8 | P-C08 | Pipeline | Transform system | §9.1 (line 1528) | ✅ Fixed — `Transform` protocol, `VariableExpansionTransform`, `TransformRegistry` in new `transforms.py` |
| 9 | P-C02 | Pipeline | Simulation mode for CodergenHandler | §4.5 (line 649) | ✅ Fixed — `backend=None` returns `[Simulated] Response for stage: {node.id}` |
| 10 | P-C10 | Pipeline | WaitHumanHandler Question/Answer interface | §4.6 (line 713) | ✅ Fixed — uses `Question(type=MULTIPLE_CHOICE, options=[Option(key, label)])` |
| 11 | A-C01 | Agent | Session terminal state → IDLE not AWAITING_INPUT | §2.3 (line 183) | ✅ Fixed — completion state changed from `AWAITING_INPUT` to `IDLE` |
| 12 | A-C04 | Agent | `edit_file` `replace_all` parameter | §3.3 (line 527) | ✅ Fixed — `replace_all: bool` parameter added (default `False`) |
| 13 | L-C02 | LLM | Provider-based routing via `provider` field | §2.2 (line 116) | ✅ Fixed — `_resolve_adapter()` checks `request.provider` first, falls back to model detection |

### Tier 3 — Improves Completeness ✅ ALL FIXED (2026-02-22, parallel agent team)

| # | Gap ID | Subsystem | Description | Spec Section | Status |
|---|---|---|---|---|---|
| 14 | P-C03 | Pipeline | Fidelity mode implementation | §5.4 (line 1139) | ✅ Fixed — 6 modes, 4-level precedence resolution, context transformation |
| 15 | P-C04 | Pipeline | ArtifactStore abstraction | §5.5 (line 1180) | ✅ Fixed — `ArtifactStore` protocol, `LocalArtifactStore` with 100KB file-backing threshold |
| 16 | P-P17/P18 | Pipeline | Emit CHECKPOINT_SAVED, PIPELINE_FAILED events | §9.6 (line 1610) | ✅ Fixed — events emitted at checkpoint saves and failure points |
| 17 | L-C06/C07 | LLM | Gemini adapter fixes (system_instruction, functionResponse) | §7.3 (line 1511) | ✅ Fixed — `system_instruction` parameter, function name in `functionResponse` |
| 18 | L-C08 | LLM | Streaming middleware | §2.3 (line 141) | ✅ Fixed — `StreamingMiddleware` protocol with `wrap_stream()`, `before_request` for stream() |
| 19 | L-C01 | LLM | `LLMClient.close()` | §2.4 (line 183) | ✅ Fixed — async `close()`, idempotent, guards on post-close operations |
| 20 | A-C05 | Agent | `ExecResult.duration_ms` | §4.2 (line 751) | ✅ Fixed — `duration_ms: int` field, populated via `time.monotonic()` |
| 21 | A-C02 | Agent | `SystemTurn` type | §2.4 (line 204) | ✅ Fixed — `SystemTurn` dataclass with `content`/`timestamp`, added to `Turn` union |

### Tier 4 — Low Priority / Optional ✅ ALL FIXED (2026-02-22, parallel agent team)

| # | Gap ID | Subsystem | Description | Spec Section | Status |
|---|---|---|---|---|---|
| 22 | P-P11 | Pipeline | `k_of_n`, `quorum`, `fail_fast`, `ignore` policies | §4.8 (line 796) | ✅ Fixed — 4 join policies + 3 error policies with `fail_fast` task cancellation |
| 23 | P-P12 | Pipeline | LLM-based evaluation in FanInHandler | §4.9 (line 856) | ✅ Fixed — LLM evaluation when `node.prompt` set, also fixed P-P13 (FAIL on empty) |
| 24 | P-C11 | Pipeline | Rearchitect ManagerLoopHandler | §4.11 (line 920) | ✅ Fixed — child pipeline mode with `child_dotfile`, observe/steer/wait cycle, legacy compat |
| 25 | P-C09 | Pipeline | HTTP server mode (optional per spec) | §9.5 (line 1590) | ✅ Fixed — stdlib async HTTP server with submit/status/events/cancel/validate endpoints |
| 26 | L-C09/C10 | LLM | `generate_object()` / `stream_object()` | §4.5/4.6 (line 959/991) | ✅ Fixed — `generate_object()` returns `GenerateResult` with `.output`, `stream_object()` async generator, `ResponseFormat` model |

---

## CLAUDE.md Corrections Applied

All 18 inaccuracies were fixed on 2026-02-21:

### Pipeline CLAUDE.md (9 fixes)
1. `conditions.py` description: `ast.parse` → custom tokenizer/parser
2. `events.py` added to module map
3. Edge routing: `priority` → `weight (higher = higher priority)`
4. Condition expressions: full rewrite — `=`/`!=`/`&&` only
5. Terminal detection: `terminal=true` → shape=Msquare / name match
6. Start node detection: `start=true` → shape=Mdiamond / name match
7. DOT example: complete rewrite with correct syntax
8. Stylesheet: `handler_type`/name glob → CSS-style selectors
9. Handler registration: `handler_type=` → `type=` in DOT

### LLM CLAUDE.md (3 fixes)
1. OpenAI adapter: "ChatCompletions API" → "Responses API"
2. Retry policy: "adapter level" → "client level"
3. `catalog.py` added to module map

### Agent CLAUDE.md (6 fixes)
1. Session lifecycle: `IDLE → RUNNING → IDLE` → `IDLE → PROCESSING → AWAITING_INPUT`
2. `apply_patch` removed from ExecutionEnvironment methods
3. `turns.py` added to module map
4. `initialize()` added to method list
5. Loop detection: removed false claim of termination
6. Loop detection wording clarified
