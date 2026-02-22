# Attractor Gap Analysis — Final Pre-Release Audit

**Date:** 2026-02-24
**Analyzed by:** Deep audit (3 parallel subsystem analyzers) with full DoD line-by-line comparison
**Previous analysis:** `gap-analysis-2026-02-23.md` (claimed 272/272 = 100% compliance)
**Specs:** `attractor-spec.md` (Pipeline), `unified-llm-spec.md` (LLM), `coding-agent-loop-spec.md` (Agent)
**Status:** 3 GAPS FOUND AND FIXED — previous analysis missed runtime-visible issues

---

## Executive Summary

| Subsystem | DoD Items | Pass | Partial | Fail | Compliance |
|-----------|:---------:|:----:|:-------:|:----:|:----------:|
| **Pipeline** | 92 | 92 | 0 | 0 | 100% |
| **LLM** | 89 | 89 | 0 | 0 | 100% |
| **Agent** | 91 | 91 | 0 | 0 | 100% |
| **Totals** | **272** | **272** | **0** | **0** | **100%** |

This audit was conducted with a different methodology than previous analyses: instead of checking "does the feature exist?", each DoD item was tested with the question "if this code ran in production, would it behave correctly?" This uncovered **2 real gaps** (one HIGH, one MEDIUM) and **1 minor test coverage gap** that the previous 100% analysis missed.

All 3 gaps have been remediated in this session.

---

## Gaps Found

### GAP 1 — HIGH: Gemini `list_dir` tool missing handler (§9.2)

| Field | Detail |
|-------|--------|
| **Spec** | coding-agent-loop-spec.md §3.2 — Gemini profile defines `list_dir` in its tool set |
| **Severity** | HIGH — runtime failure when Gemini invokes `list_dir` |
| **Root Cause** | `GeminiProfile` declared `list_dir` in `tool_definitions` but no handler existed in `core_tools.py` and no entry in `Session._build_registry._tool_map` |
| **Impact** | Gemini model invoking `list_dir` would receive an "unknown tool" error at runtime |
| **Fix Applied** | Added `list_dir_tool` handler + `LIST_DIR_DEF` in `core_tools.py`, wired into `_tool_map` in `session.py`, added truncation limits in `truncation.py`, updated `gemini_profile.py` to import from `core_tools.py` |
| **Tests Added** | 6 tests in `test_tools.py::TestListDir`, 2 tests in `test_session.py::TestGeminiListDir` |

**Files modified:**
- `src/attractor/agent/tools/core_tools.py` — Added `list_dir_tool()` handler and `LIST_DIR_DEF`
- `src/attractor/agent/session.py` — Added `"list_dir"` to `_tool_map`
- `src/attractor/agent/truncation.py` — Added `list_dir` to `char_limits` (20K), `truncation_modes` (tail), `line_limits` (500)
- `src/attractor/agent/profiles/gemini_profile.py` — Replaced local `_LIST_DIR_DEF` with import from `core_tools`

### GAP 2 — MEDIUM: Anthropic shell timeout should be 120s (§9.4 / §3.5)

| Field | Detail |
|-------|--------|
| **Spec** | coding-agent-loop-spec.md §3.5 — "shell (bash execution, 120s default timeout per Claude Code convention)" |
| **Severity** | MEDIUM — commands that need >10s would time out prematurely on Anthropic |
| **Root Cause** | All profiles shared a hardcoded 10s default timeout via `SessionConfig.default_command_timeout_ms = 10_000` |
| **Impact** | Anthropic profile should use 120s per Claude Code conventions; long-running builds/tests would fail |
| **Fix Applied** | Added `default_timeout_ms` property to `ProviderProfile` protocol, set to 120_000 in `AnthropicProfile`, 10_000 in others. `SessionConfig.default_command_timeout_ms` changed to `int | None = None` (None = use profile default). Session uses profile timeout when config is None. |
| **Tests Added** | 5 tests in `test_session.py::TestProfileTimeout` |

**Files modified:**
- `src/attractor/agent/profiles/base.py` — Added `default_timeout_ms` to `ProviderProfile` protocol
- `src/attractor/agent/profiles/anthropic_profile.py` — Returns `120_000`
- `src/attractor/agent/profiles/openai_profile.py` — Returns `10_000`
- `src/attractor/agent/profiles/gemini_profile.py` — Returns `10_000`
- `src/attractor/agent/session.py` — Uses profile timeout when `SessionConfig.default_command_timeout_ms is None`

### GAP 3 — LOW: No test for DOT comment stripping (§11.1)

| Field | Detail |
|-------|--------|
| **Spec** | attractor-spec.md §11.1 — DOT parser handles comments |
| **Severity** | LOW — functionality works (delegated to pydot), but no explicit test guardrail |
| **Root Cause** | Comment stripping is handled by the `pydot` library, so no custom code was needed. However, no test verified this behavior, leaving it vulnerable to future parser changes. |
| **Impact** | A pydot version change could silently break comment handling |
| **Fix Applied** | Added 3 tests verifying `//` comments, `/* */` block comments, and mixed comments in DOT files |
| **Tests Added** | 3 tests in `test_parser.py::TestDotComments` |

**File modified:**
- `tests/pipeline/test_parser.py` — Added `TestDotComments` class with 3 test cases

---

## Why Previous Analysis Missed These

1. **GAP 1 (list_dir handler):** The previous analysis verified that `GeminiProfile.tool_definitions` includes `list_dir` — which it does. It did not trace the full execution path to confirm that `Session._build_registry._tool_map` has a corresponding handler. The profile declares the tool, but the session never wires it up.

2. **GAP 2 (Anthropic timeout):** The previous analysis checked that shell timeout enforcement exists (it does, in `loop.py:158-161`). It did not compare the default value (10s) against the spec's per-provider requirement (120s for Anthropic).

3. **GAP 3 (comment test):** The previous analysis confirmed comment stripping works via the pydot library but didn't check for explicit test coverage.

---

## Verification

### Test Results

```
$ pytest -m "not smoke" -v
===================== 1285 passed, 29 deselected in 9.44s ======================
```

All 1285 non-smoke tests pass. The 29 deselected are smoke tests requiring real API keys.

### New Tests Summary

| Gap | Test File | Test Class | Count |
|-----|-----------|------------|:-----:|
| GAP 1 | `tests/agent/test_tools.py` | `TestListDir` | 6 |
| GAP 1 | `tests/agent/test_session.py` | `TestGeminiListDir` | 2 |
| GAP 2 | `tests/agent/test_session.py` | `TestProfileTimeout` | 5 |
| GAP 3 | `tests/pipeline/test_parser.py` | `TestDotComments` | 3 |
| **Total** | | | **16** |

---

## Files Modified Summary

| File | Change |
|------|--------|
| `specs/gaps/gap-analysis-2026-02-24.md` | This document |
| `src/attractor/agent/tools/core_tools.py` | Added `list_dir_tool` handler and `LIST_DIR_DEF` |
| `src/attractor/agent/session.py` | Wired `list_dir` in `_tool_map`; profile-based timeout in `LoopConfig` |
| `src/attractor/agent/truncation.py` | Added `list_dir` truncation limits (20K chars, tail mode, 500 lines) |
| `src/attractor/agent/profiles/base.py` | Added `default_timeout_ms` to `ProviderProfile` protocol |
| `src/attractor/agent/profiles/anthropic_profile.py` | Set `default_timeout_ms = 120_000` |
| `src/attractor/agent/profiles/openai_profile.py` | Set `default_timeout_ms = 10_000` |
| `src/attractor/agent/profiles/gemini_profile.py` | Import `LIST_DIR_DEF` from core_tools; set `default_timeout_ms = 10_000` |
| `tests/agent/test_tools.py` | Added `TestListDir` (6 tests) |
| `tests/agent/test_session.py` | Added `TestGeminiListDir` (2 tests) + `TestProfileTimeout` (5 tests) |
| `tests/pipeline/test_parser.py` | Added `TestDotComments` (3 tests) |

---

## Comparison with Previous Analysis

| Metric | 2026-02-23 (prev.) | 2026-02-24 (this) | Delta |
|--------|:------------------:|:-----------------:|:-----:|
| Total DoD items | 272 | 272 | — |
| PASS | 272 | 272 | 0 |
| PARTIAL | 0 | 0 | 0 |
| FAIL | 0 | 0 | 0 |
| Compliance | 100% | 100% | 0 |
| Total tests | 1,269 | 1,285 | +16 |

---

## Notes

- **Methodology shift:** This audit focused on "runtime correctness" rather than "feature existence." Each DoD item was traced through the full execution path (profile → session → registry → handler → environment) to confirm end-to-end behavior.

- **GAP 1 is the most impactful.** It would have caused a visible runtime error the first time a Gemini model tried to use `list_dir`. The fix is straightforward — the `ExecutionEnvironment.list_directory()` method already existed; only the tool handler and registry wiring were missing.

- **GAP 2 is a correctness issue.** A 10s default timeout is too aggressive for builds, test suites, and other long-running commands that Claude Code users routinely execute. The 120s default matches Claude Code's actual behavior.

- **Integration smoke tests remain manual validation items.** §8.10, §9.13, and §11.13 require real API keys across 3 providers. They are excluded from compliance percentages.
