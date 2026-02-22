"""Tests for the HTTP server mode (spec §9.5)."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.models import PipelineContext
from attractor.pipeline.server import (
    PipelineRun,
    PipelineRunStatus,
    PipelineServer,
)


# ---------------------------------------------------------------------------
# Valid DOT source used across tests
# ---------------------------------------------------------------------------
VALID_DOT = """\
digraph test_pipeline {
    start [shape=Mdiamond]
    work [type="codergen" prompt="do work"]
    done [shape=Msquare]

    start -> work
    work -> done
}
"""

INVALID_DOT = "this is not a valid DOT file {{{{"

INVALID_PIPELINE_DOT = """\
digraph bad {
    a [type="codergen"]
    b [type="codergen"]
    a -> b
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def send_request(
    server: PipelineServer,
    method: str,
    path: str,
    body: str = "",
) -> tuple[int, dict]:
    """Send an HTTP request to the server and return (status_code, json_body).

    Opens a real TCP connection to the server's bound port so we test
    the full request parsing and response generation path.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", server._port)
    body_bytes = body.encode("utf-8")
    request = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Content-Type: application/json\r\n"
        f"\r\n"
    )
    writer.write(request.encode("utf-8") + body_bytes)
    await writer.drain()

    # Read the full response
    response_data = await asyncio.wait_for(reader.read(65536), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    response_str = response_data.decode("utf-8")

    # Parse status code from status line
    status_line = response_str.split("\r\n")[0]
    status_code = int(status_line.split(" ")[1])

    # Parse body (after empty line separator)
    body_str = response_str.split("\r\n\r\n", 1)[1]
    return status_code, json.loads(body_str)


@pytest.fixture
async def server():
    """Create and start a PipelineServer on a random available port."""
    srv = PipelineServer(host="127.0.0.1", port=0)
    await srv.start()
    # Get the actual port assigned
    assert srv._server is not None
    srv._port = srv._server.sockets[0].getsockname()[1]
    yield srv
    await srv.stop()


# ---------------------------------------------------------------------------
# POST /pipelines — Submit pipeline
# ---------------------------------------------------------------------------


class TestSubmitPipeline:
    """Tests for the POST /pipelines endpoint."""

    async def test_submit_valid_pipeline(self, server: PipelineServer) -> None:
        """Submitting a valid DOT returns 202 with a run ID."""
        with patch.object(
            PipelineEngine, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_ctx = PipelineContext()
            mock_ctx.set("outcome", "success")
            mock_run.return_value = mock_ctx

            status, body = await send_request(
                server, "POST", "/pipelines", json.dumps({"dot": VALID_DOT})
            )

        assert status == 202
        assert "id" in body
        assert body["status"] == "running"
        assert len(body["id"]) == 12

    async def test_submit_invalid_json(self, server: PipelineServer) -> None:
        """Submitting non-JSON returns 400."""
        status, body = await send_request(
            server, "POST", "/pipelines", "not json"
        )
        assert status == 400
        assert "error" in body
        assert "Invalid JSON" in body["error"]

    async def test_submit_missing_dot_field(self, server: PipelineServer) -> None:
        """Submitting JSON without 'dot' field returns 400."""
        status, body = await send_request(
            server, "POST", "/pipelines", json.dumps({"other": "data"})
        )
        assert status == 400
        assert "Missing" in body["error"]

    async def test_submit_unparseable_dot(self, server: PipelineServer) -> None:
        """Submitting invalid DOT syntax returns 400."""
        status, body = await send_request(
            server, "POST", "/pipelines", json.dumps({"dot": INVALID_DOT})
        )
        assert status == 400
        assert "error" in body

    async def test_submit_pipeline_with_validation_errors(
        self, server: PipelineServer
    ) -> None:
        """Submitting DOT that fails validation returns 400 with details."""
        # A pipeline with no start node should fail validation
        dot = 'digraph bad { a [type="codergen"]; b [type="codergen"]; a -> b }'
        status, body = await send_request(
            server, "POST", "/pipelines", json.dumps({"dot": dot})
        )
        assert status == 400
        assert "error" in body

    async def test_submit_empty_body(self, server: PipelineServer) -> None:
        """Submitting an empty body returns 400."""
        status, body = await send_request(server, "POST", "/pipelines", "")
        assert status == 400
        assert "error" in body


# ---------------------------------------------------------------------------
# GET /pipelines/{id} — Status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for the GET /pipelines/{id} endpoint."""

    async def test_get_status_not_found(self, server: PipelineServer) -> None:
        """Requesting a non-existent run ID returns 404."""
        status, body = await send_request(
            server, "GET", "/pipelines/nonexistent"
        )
        assert status == 404
        assert "not found" in body["error"]

    async def test_get_status_running(self, server: PipelineServer) -> None:
        """Status shows 'running' for an in-progress pipeline."""
        # Insert a run directly
        run = PipelineRun(id="test123", status=PipelineRunStatus.RUNNING)
        server.runs["test123"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/test123"
        )
        assert status == 200
        assert body["id"] == "test123"
        assert body["status"] == "running"

    async def test_get_status_completed(self, server: PipelineServer) -> None:
        """Status shows 'completed' for a finished pipeline."""
        run = PipelineRun(
            id="done456",
            status=PipelineRunStatus.COMPLETED,
            pipeline_name="my_pipeline",
            completed_at=time.time(),
        )
        server.runs["done456"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/done456"
        )
        assert status == 200
        assert body["status"] == "completed"
        assert body["pipeline_name"] == "my_pipeline"
        assert body["completed_at"] is not None

    async def test_get_status_failed(self, server: PipelineServer) -> None:
        """Status shows 'failed' with error message for a failed pipeline."""
        run = PipelineRun(
            id="fail789",
            status=PipelineRunStatus.FAILED,
            error="Handler not found",
            completed_at=time.time(),
        )
        server.runs["fail789"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/fail789"
        )
        assert status == 200
        assert body["status"] == "failed"
        assert body["error"] == "Handler not found"


# ---------------------------------------------------------------------------
# GET /pipelines/{id}/context — Context
# ---------------------------------------------------------------------------


class TestGetContext:
    """Tests for the GET /pipelines/{id}/context endpoint."""

    async def test_get_context_not_found(self, server: PipelineServer) -> None:
        """Requesting context for unknown run returns 404."""
        status, body = await send_request(
            server, "GET", "/pipelines/unknown/context"
        )
        assert status == 404

    async def test_get_context_running(self, server: PipelineServer) -> None:
        """Context for a running pipeline returns empty with message."""
        run = PipelineRun(id="ctx1", status=PipelineRunStatus.RUNNING)
        server.runs["ctx1"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/ctx1/context"
        )
        assert status == 200
        assert body["status"] == "running"
        assert body["context"] == {}
        assert "still running" in body["message"]

    async def test_get_context_completed(self, server: PipelineServer) -> None:
        """Context for a completed pipeline returns the full context dict."""
        run = PipelineRun(
            id="ctx2",
            status=PipelineRunStatus.COMPLETED,
            context={"outcome": "success", "result": "done"},
        )
        server.runs["ctx2"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/ctx2/context"
        )
        assert status == 200
        assert body["context"]["outcome"] == "success"
        assert body["context"]["result"] == "done"


# ---------------------------------------------------------------------------
# GET /pipelines/{id}/events — Events
# ---------------------------------------------------------------------------


class TestGetEvents:
    """Tests for the GET /pipelines/{id}/events endpoint."""

    async def test_get_events_not_found(self, server: PipelineServer) -> None:
        """Requesting events for unknown run returns 404."""
        status, body = await send_request(
            server, "GET", "/pipelines/unknown/events"
        )
        assert status == 404

    async def test_get_events_with_data(self, server: PipelineServer) -> None:
        """Events endpoint returns collected events."""
        run = PipelineRun(
            id="ev1",
            status=PipelineRunStatus.COMPLETED,
            events=[
                {
                    "type": "pipeline_start",
                    "node_name": "",
                    "pipeline_name": "test",
                    "timestamp": 1000.0,
                    "data": {},
                },
                {
                    "type": "node_start",
                    "node_name": "work",
                    "pipeline_name": "test",
                    "timestamp": 1001.0,
                    "data": {"handler_type": "codergen"},
                },
            ],
        )
        server.runs["ev1"] = run

        status, body = await send_request(
            server, "GET", "/pipelines/ev1/events"
        )
        assert status == 200
        assert len(body["events"]) == 2
        assert body["events"][0]["type"] == "pipeline_start"
        assert body["events"][1]["node_name"] == "work"


# ---------------------------------------------------------------------------
# POST /pipelines/{id}/cancel — Cancel
# ---------------------------------------------------------------------------


class TestCancelPipeline:
    """Tests for the POST /pipelines/{id}/cancel endpoint."""

    async def test_cancel_not_found(self, server: PipelineServer) -> None:
        """Cancelling unknown run returns 404."""
        status, body = await send_request(
            server, "POST", "/pipelines/nope/cancel"
        )
        assert status == 404

    async def test_cancel_not_running(self, server: PipelineServer) -> None:
        """Cancelling a completed run returns 409."""
        run = PipelineRun(
            id="done1", status=PipelineRunStatus.COMPLETED
        )
        server.runs["done1"] = run

        status, body = await send_request(
            server, "POST", "/pipelines/done1/cancel"
        )
        assert status == 409
        assert "not running" in body["error"]

    async def test_cancel_running_pipeline(self, server: PipelineServer) -> None:
        """Cancelling a running pipeline sets status to cancelled."""
        # Create a long-running mock task
        async def long_task() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(long_task())
        run = PipelineRun(
            id="cancel1",
            status=PipelineRunStatus.RUNNING,
            task=task,
        )
        server.runs["cancel1"] = run

        status, body = await send_request(
            server, "POST", "/pipelines/cancel1/cancel"
        )
        assert status == 200
        assert body["status"] == "cancelled"

        # Verify the task was cancelled
        await asyncio.sleep(0.05)
        assert task.cancelled()


# ---------------------------------------------------------------------------
# POST /validate — Validate pipeline
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for the POST /validate endpoint."""

    async def test_validate_valid_pipeline(self, server: PipelineServer) -> None:
        """Validating a correct DOT returns valid=True."""
        status, body = await send_request(
            server, "POST", "/validate", json.dumps({"dot": VALID_DOT})
        )
        assert status == 200
        assert body["valid"] is True
        assert body["errors"] == []
        assert "pipeline_name" in body

    async def test_validate_invalid_dot_syntax(
        self, server: PipelineServer
    ) -> None:
        """Validating unparseable DOT returns valid=False with parse error."""
        status, body = await send_request(
            server, "POST", "/validate", json.dumps({"dot": INVALID_DOT})
        )
        assert status == 200
        assert body["valid"] is False
        assert len(body["errors"]) > 0

    async def test_validate_missing_dot_field(
        self, server: PipelineServer
    ) -> None:
        """Validating without 'dot' field returns 400."""
        status, body = await send_request(
            server, "POST", "/validate", json.dumps({"source": "..."})
        )
        assert status == 400
        assert "Missing" in body["error"]

    async def test_validate_invalid_json(self, server: PipelineServer) -> None:
        """Validating with non-JSON body returns 400."""
        status, body = await send_request(
            server, "POST", "/validate", "not json"
        )
        assert status == 400

    async def test_validate_pipeline_with_errors(
        self, server: PipelineServer
    ) -> None:
        """Validating a pipeline with structural errors returns valid=False."""
        dot = 'digraph bad { a [type="codergen"]; b [type="codergen"]; a -> b }'
        status, body = await send_request(
            server, "POST", "/validate", json.dumps({"dot": dot})
        )
        assert status == 200
        assert body["valid"] is False
        assert len(body["errors"]) > 0


# ---------------------------------------------------------------------------
# Routing — 404 for unknown paths
# ---------------------------------------------------------------------------


class TestRouting:
    """Tests for general routing and 404 handling."""

    async def test_unknown_path_returns_404(
        self, server: PipelineServer
    ) -> None:
        """Unknown paths return 404."""
        status, body = await send_request(server, "GET", "/unknown")
        assert status == 404
        assert "Not found" in body["error"]

    async def test_wrong_method_returns_404(
        self, server: PipelineServer
    ) -> None:
        """Wrong HTTP method for a known path returns 404."""
        status, body = await send_request(server, "DELETE", "/pipelines")
        assert status == 404

    async def test_get_root_returns_404(
        self, server: PipelineServer
    ) -> None:
        """GET / returns 404 (no root endpoint defined)."""
        status, body = await send_request(server, "GET", "/")
        assert status == 404


# ---------------------------------------------------------------------------
# Pipeline execution integration
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    """Tests for async pipeline execution tracking."""

    async def test_pipeline_execution_completes(
        self, server: PipelineServer
    ) -> None:
        """A submitted pipeline eventually reaches completed status."""
        with patch.object(
            PipelineEngine, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_ctx = PipelineContext()
            mock_ctx.set("outcome", "success")
            mock_run.return_value = mock_ctx

            status, body = await send_request(
                server, "POST", "/pipelines", json.dumps({"dot": VALID_DOT})
            )
            assert status == 202
            run_id = body["id"]

            # Wait for the background task to complete
            run = server.runs[run_id]
            assert run.task is not None
            await asyncio.wait_for(run.task, timeout=5.0)

            # Check final status
            status2, body2 = await send_request(
                server, "GET", f"/pipelines/{run_id}"
            )
            assert status2 == 200
            assert body2["status"] == "completed"

    async def test_pipeline_execution_failure(
        self, server: PipelineServer
    ) -> None:
        """A pipeline that raises an exception is marked as failed."""
        with patch.object(
            PipelineEngine, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = RuntimeError("Handler exploded")

            status, body = await send_request(
                server, "POST", "/pipelines", json.dumps({"dot": VALID_DOT})
            )
            assert status == 202
            run_id = body["id"]

            # Wait for the background task
            run = server.runs[run_id]
            assert run.task is not None
            await asyncio.wait_for(run.task, timeout=5.0)

            status2, body2 = await send_request(
                server, "GET", f"/pipelines/{run_id}"
            )
            assert status2 == 200
            assert body2["status"] == "failed"
            assert "Handler exploded" in body2["error"]


# ---------------------------------------------------------------------------
# PipelineRun model
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Tests for the PipelineRun dataclass."""

    def test_to_dict_includes_required_fields(self) -> None:
        """to_dict returns all expected fields."""
        run = PipelineRun(
            id="abc",
            status=PipelineRunStatus.RUNNING,
            pipeline_name="test",
        )
        d = run.to_dict()
        assert d["id"] == "abc"
        assert d["status"] == "running"
        assert d["pipeline_name"] == "test"
        assert d["error"] is None
        assert "created_at" in d
        assert d["completed_at"] is None

    def test_to_dict_completed_includes_timestamps(self) -> None:
        """Completed runs include both timestamps."""
        now = time.time()
        run = PipelineRun(
            id="xyz",
            status=PipelineRunStatus.COMPLETED,
            created_at=now - 10,
            completed_at=now,
        )
        d = run.to_dict()
        assert d["completed_at"] is not None
        assert d["completed_at"] > d["created_at"]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    """Tests for server start/stop."""

    async def test_start_and_stop(self) -> None:
        """Server can be started and stopped without errors."""
        srv = PipelineServer(host="127.0.0.1", port=0)
        await srv.start()
        assert srv._server is not None
        await srv.stop()
        assert srv._server is None

    async def test_stop_cancels_running_tasks(self) -> None:
        """Stopping the server cancels any running pipeline tasks."""
        srv = PipelineServer(host="127.0.0.1", port=0)
        await srv.start()

        async def long_task() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(long_task())
        run = PipelineRun(
            id="running1",
            status=PipelineRunStatus.RUNNING,
            task=task,
        )
        srv.runs["running1"] = run

        await srv.stop()
        await asyncio.sleep(0.05)
        assert task.cancelled()
