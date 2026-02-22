"""Pipeline subsystem integration smoke tests (spec §11.13).

Requires a real ANTHROPIC_API_KEY.  Run with:

    pytest -m smoke tests/pipeline/test_smoke_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attractor.llm.client import LLMClient
from attractor.pipeline.models import PipelineContext, PipelineNode
from attractor.pipeline.parser import parse_dot_string
from attractor.pipeline.validator import has_errors, validate_pipeline

SMOKE_MODEL = "claude-sonnet-4-6"

# §11.13 DOT pipeline — a minimal plan → implement → review workflow.
# Uses the simple LLM backend (not a full agent session) to keep cost down.
DOT_PIPELINE = """\
digraph test_pipeline {
    graph [goal="Create a hello world Python script"]

    start       [shape=Mdiamond]
    plan        [shape=box prompt="Plan how to create a hello world script" model="claude-sonnet-4-6"]
    implement   [shape=box prompt="Write the code based on the plan" goal_gate=true retry_target="plan" model="claude-sonnet-4-6"]
    review      [shape=box prompt="Review the code for correctness" model="claude-sonnet-4-6"]
    done        [shape=Msquare]

    start -> plan
    plan -> implement
    implement -> review  [condition="outcome = success"]
    implement -> plan    [condition="outcome = fail" label="Retry"]
    review -> done       [condition="outcome = success"]
    review -> implement  [condition="outcome = fail" label="Fix"]
}
"""

pytestmark = pytest.mark.smoke


class _SimpleLLMBackend:
    """Lightweight backend that calls the LLM directly (no agent session).

    This avoids spinning up a full agent with tool-use loops for each
    pipeline node — much cheaper and faster while still exercising the
    real Anthropic API.
    """

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model

    async def run(
        self,
        node: PipelineNode,
        prompt: str,
        context: PipelineContext,
    ) -> str:
        result = await self._client.generate(prompt=prompt, model=self._model)
        return result.text


class TestPipelineSmoke:
    """Smoke tests for the pipeline engine."""

    # §11.13 steps 1-2 — parse DOT and validate
    def test_parse_dot_pipeline(self) -> None:
        pipeline = parse_dot_string(DOT_PIPELINE, name="test_pipeline")

        assert len(pipeline.nodes) == 5, (
            f"Expected 5 nodes, got {len(pipeline.nodes)}"
        )
        assert len(pipeline.edges) == 6, (
            f"Expected 6 edges, got {len(pipeline.edges)}"
        )
        assert pipeline.start_node is not None, "Expected a start node"

        findings = validate_pipeline(pipeline)
        assert not has_errors(findings), (
            f"Validation errors: {[str(f) for f in findings]}"
        )

    # §11.13 steps 3-5 — execute pipeline end-to-end with real LLM,
    # check outcome and artifacts
    async def test_execute_pipeline(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        from attractor.pipeline.engine import PipelineEngine
        from attractor.pipeline.handlers import (
            CodergenHandler,
            create_default_registry,
        )

        pipeline = parse_dot_string(DOT_PIPELINE, name="test_pipeline")
        client = LLMClient.from_env()
        backend = _SimpleLLMBackend(client, SMOKE_MODEL)

        registry = create_default_registry(pipeline=pipeline)
        registry.register(
            "codergen", CodergenHandler(backend=backend)
        )

        logs_root = tmp_path / "logs"
        engine = PipelineEngine(
            registry=registry,
            logs_root=logs_root,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        context = await engine.run(pipeline)

        # The engine should reach the terminal node successfully
        assert context is not None, "Expected a non-None pipeline context"
        # The last node executed should be 'done' (the exit node)
        completed = context.get("_completed_nodes", [])
        assert "done" in completed or context.get("outcome") == "success", (
            f"Expected pipeline to reach exit; completed={completed}"
        )

    # §11.13 step 4 — verify status artifacts written per node
    async def test_artifacts_written(
        self, requires_anthropic_key: None, tmp_path: Path
    ) -> None:
        from attractor.pipeline.engine import PipelineEngine
        from attractor.pipeline.handlers import (
            CodergenHandler,
            create_default_registry,
        )

        pipeline = parse_dot_string(DOT_PIPELINE, name="test_pipeline")
        client = LLMClient.from_env()
        backend = _SimpleLLMBackend(client, SMOKE_MODEL)

        registry = create_default_registry(pipeline=pipeline)
        registry.register(
            "codergen", CodergenHandler(backend=backend)
        )

        logs_root = tmp_path / "logs"
        engine = PipelineEngine(
            registry=registry,
            logs_root=logs_root,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        await engine.run(pipeline)

        # Each codergen node should have a status.json artifact
        for node_name in ("plan", "implement", "review"):
            status_path = logs_root / node_name / "status.json"
            assert status_path.exists(), (
                f"Missing status.json for node '{node_name}'"
            )
