# CLAUDE.md — Test Suite

Tests mirror the source layout: `tests/agent/`, `tests/llm/`, `tests/pipeline/`. All tests use pytest with the `pytest-asyncio` plugin in auto mode.

## Running Tests

```bash
pytest -v                                 # All tests, verbose
pytest tests/agent/ -v                    # Agent tests only
pytest tests/pipeline/test_engine.py -v   # Single file
pytest -k "test_session" -v               # By name pattern
pytest --cov=src/attractor --cov-report=html  # With coverage
```

## Conventions

- **File naming**: `test_<module>.py` matches the source module it tests.
- **Test classes**: Group related tests in `class TestXxx`. Each class tests one unit (a function, a class, or a behavior).
- **Async tests**: Decorated with `@pytest.mark.asyncio`. The `asyncio_mode = "auto"` config in `pyproject.toml` auto-detects most cases.
- **Fixtures**: Use `@pytest.fixture` for setup. Common fixtures include `tmp_path` (built-in) for filesystem tests and mock LLM clients for session tests.
- **Mocking strategy**: External dependencies (LLM APIs, file system, subprocesses) are mocked. Internal logic is tested directly without mocks.
- **No conftest.py**: Fixtures are defined locally in each test file. Add a `conftest.py` if shared fixtures are needed across files in a subdirectory.

## Test Inventory

| File | Tests |
|---|---|
| `agent/test_session.py` | Session lifecycle, text responses, tool call dispatch, event flow, follow-up queue |
| `agent/test_tools.py` | Each core tool (read/write/edit/shell/grep/glob) against a temp filesystem |
| `agent/test_environment.py` | `LocalExecutionEnvironment` file and command operations |
| `agent/test_loop_detection.py` | Pattern detection for repeating tool call sequences |
| `agent/test_truncation.py` | Two-stage truncation with various content sizes |
| `llm/test_models.py` | Message, Request, Response construction and serialization |
| `llm/test_client.py` | Adapter detection, routing, middleware application, tool loops, retry |
| `pipeline/test_engine.py` | Linear/branching pipelines, conditions, failures, checkpointing, resume |
| `pipeline/test_parser.py` | DOT parsing, attribute extraction, start/terminal detection |
| `pipeline/test_validator.py` | All static validation checks (missing nodes, bad edges, unreachable) |
| `pipeline/test_conditions.py` | Expression evaluation with all operators and variable resolution |
| `pipeline/test_state.py` | Checkpoint save/load/list, timestamp ordering |

## Writing New Tests

- Place the test file in the matching subdirectory under `tests/`.
- Mock the LLM client — never make real API calls in tests.
- Use `tmp_path` for any test that touches the filesystem.
- Test both success and failure paths. Include edge cases (empty input, missing fields, boundary values).
- Target 80%+ coverage for new code.
- Suppress known warnings in `pyproject.toml` `filterwarnings` if needed (see pyparsing example).
