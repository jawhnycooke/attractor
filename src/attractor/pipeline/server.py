"""HTTP server mode for pipeline execution.

Exposes the pipeline engine as an HTTP service for web-based management,
remote interaction, and integration with external systems per spec §9.5.

Uses only Python standard library (``asyncio``, ``http``, ``json``,
``urllib.parse``) — no third-party HTTP framework required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http import HTTPStatus
from typing import Any

from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.events import PipelineEvent, PipelineEventEmitter, PipelineEventType
from attractor.pipeline.parser import ParseError, parse_dot_string
from attractor.pipeline.validator import ValidationLevel, validate_pipeline

logger = logging.getLogger(__name__)


class PipelineRunStatus(str, Enum):
    """Status of a submitted pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineRun:
    """Tracks a single pipeline execution.

    Attributes:
        id: Unique run identifier.
        status: Current execution status.
        dot_source: The original DOT source submitted.
        pipeline_name: Name extracted from the parsed pipeline.
        context: Final pipeline context (populated on completion).
        error: Error message if the run failed.
        events: Collected pipeline events for this run.
        task: The asyncio task running the pipeline.
        created_at: UNIX timestamp when the run was submitted.
        completed_at: UNIX timestamp when the run finished.
    """

    id: str
    status: PipelineRunStatus = PipelineRunStatus.PENDING
    dot_source: str = ""
    pipeline_name: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize run state to a JSON-compatible dict."""
        return {
            "id": self.id,
            "status": self.status.value,
            "pipeline_name": self.pipeline_name,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class PipelineServer:
    """Async HTTP server wrapping PipelineEngine.

    Manages concurrent pipeline runs, tracks their status, and exposes
    a REST API for submission, status queries, and validation.

    Args:
        engine: The pipeline engine to use for execution.
        host: Bind address (default ``"127.0.0.1"``).
        port: Bind port (default ``8080``).
    """

    def __init__(
        self,
        engine: PipelineEngine | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self._engine = engine or PipelineEngine()
        self._host = host
        self._port = port
        self._runs: dict[str, PipelineRun] = {}
        self._server: asyncio.Server | None = None

    @property
    def runs(self) -> dict[str, PipelineRun]:
        """Return the mapping of run IDs to PipelineRun objects."""
        return self._runs

    async def start(self) -> None:
        """Start the HTTP server."""
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        logger.info("Pipeline server listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the HTTP server and cancel all running pipelines."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Cancel running pipelines
        for run in self._runs.values():
            if run.task and not run.task.done():
                run.task.cancel()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            request_line, _headers, body = await self._read_request(reader)
            if request_line is None:
                writer.close()
                await writer.wait_closed()
                return

            method, path, _ = request_line.split(" ", 2)
            status, response_body = await self._route(method, path, body)
            await self._send_response(writer, status, response_body)
        except Exception as exc:
            logger.error("Connection handler error: %s", exc)
            try:
                error_body = json.dumps({"error": "Internal server error"})
                await self._send_response(
                    writer, HTTPStatus.INTERNAL_SERVER_ERROR, error_body
                )
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> tuple[str | None, dict[str, str], str]:
        """Parse an HTTP request from the stream.

        Returns:
            Tuple of (request_line, headers_dict, body_string).
            request_line is None if the connection was closed.
        """
        # Read request line
        try:
            request_line_bytes = await asyncio.wait_for(
                reader.readline(), timeout=30.0
            )
        except (asyncio.TimeoutError, ConnectionError):
            return None, {}, ""

        if not request_line_bytes:
            return None, {}, ""

        request_line = request_line_bytes.decode("utf-8").strip()
        if not request_line:
            return None, {}, ""

        # Read headers
        headers: dict[str, str] = {}
        while True:
            line_bytes = await reader.readline()
            line = line_bytes.decode("utf-8").strip()
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        # Read body if content-length is present
        body = ""
        content_length = int(headers.get("content-length", "0"))
        if content_length > 0:
            body_bytes = await reader.readexactly(content_length)
            body = body_bytes.decode("utf-8")

        return request_line, headers, body

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: str,
    ) -> None:
        """Write an HTTP response to the stream."""
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()

    async def _route(
        self, method: str, path: str, body: str
    ) -> tuple[HTTPStatus, str]:
        """Route an HTTP request to the appropriate handler.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path.
            body: Request body string.

        Returns:
            Tuple of (HTTP status, JSON response body).
        """
        # Strip query string
        path = path.split("?")[0].rstrip("/")

        # POST /pipelines — submit a new pipeline
        if method == "POST" and path == "/pipelines":
            return await self._handle_submit(body)

        # POST /validate — validate a DOT file
        if method == "POST" and path == "/validate":
            return await self._handle_validate(body)

        # GET /pipelines/{id} — get pipeline status
        if method == "GET" and path.startswith("/pipelines/"):
            parts = path.split("/")
            if len(parts) == 3:
                return self._handle_status(parts[2])
            if len(parts) == 4 and parts[3] == "context":
                return self._handle_context(parts[2])
            if len(parts) == 4 and parts[3] == "events":
                return self._handle_events(parts[2])

        # POST /pipelines/{id}/cancel — cancel a running pipeline
        if method == "POST" and path.startswith("/pipelines/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] == "cancel":
                return self._handle_cancel(parts[2])

        return (
            HTTPStatus.NOT_FOUND,
            json.dumps({"error": f"Not found: {method} {path}"}),
        )

    async def _handle_submit(self, body: str) -> tuple[HTTPStatus, str]:
        """Handle POST /pipelines — submit a pipeline for execution."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({"error": "Invalid JSON body"}),
            )

        dot_source = data.get("dot")
        if not dot_source or not isinstance(dot_source, str):
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({"error": "Missing or invalid 'dot' field"}),
            )

        # Parse the DOT source
        try:
            pipeline = parse_dot_string(dot_source)
        except ParseError as exc:
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({"error": f"Parse error: {exc}"}),
            )

        # Validate the pipeline
        errors = validate_pipeline(pipeline)
        validation_errors = [
            e for e in errors if e.level == ValidationLevel.ERROR
        ]
        if validation_errors:
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({
                    "error": "Pipeline validation failed",
                    "details": [str(e) for e in validation_errors],
                }),
            )

        # Create and track the run
        run_id = uuid.uuid4().hex[:12]
        run = PipelineRun(
            id=run_id,
            status=PipelineRunStatus.RUNNING,
            dot_source=dot_source,
            pipeline_name=pipeline.name,
        )
        self._runs[run_id] = run

        # Set up event collection
        emitter = PipelineEventEmitter()
        for event_type in PipelineEventType:
            emitter.on(event_type, self._make_event_collector(run))

        # Create an engine with the event emitter for this run
        engine = PipelineEngine(event_emitter=emitter)

        # Start async execution
        run.task = asyncio.create_task(
            self._execute_pipeline(run, engine, pipeline)
        )

        return (
            HTTPStatus.ACCEPTED,
            json.dumps({"id": run_id, "status": run.status.value}),
        )

    async def _execute_pipeline(
        self,
        run: PipelineRun,
        engine: PipelineEngine,
        pipeline: Any,
    ) -> None:
        """Execute a pipeline and update the run state."""
        try:
            ctx = await engine.run(pipeline)
            run.status = PipelineRunStatus.COMPLETED
            run.context = ctx.to_dict()
            run.completed_at = time.time()
        except asyncio.CancelledError:
            run.status = PipelineRunStatus.CANCELLED
            run.completed_at = time.time()
        except Exception as exc:
            run.status = PipelineRunStatus.FAILED
            run.error = str(exc)
            run.completed_at = time.time()
            logger.error("Pipeline run %s failed: %s", run.id, exc)

    def _make_event_collector(
        self, run: PipelineRun
    ) -> Any:
        """Create an event callback that appends events to a run."""
        async def collector(event: PipelineEvent) -> None:
            run.events.append({
                "type": event.type.value,
                "node_name": event.node_name,
                "pipeline_name": event.pipeline_name,
                "timestamp": event.timestamp,
                "data": event.data,
            })
        return collector

    async def _handle_validate(self, body: str) -> tuple[HTTPStatus, str]:
        """Handle POST /validate — validate a DOT file without executing."""
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({"error": "Invalid JSON body"}),
            )

        dot_source = data.get("dot")
        if not dot_source or not isinstance(dot_source, str):
            return (
                HTTPStatus.BAD_REQUEST,
                json.dumps({"error": "Missing or invalid 'dot' field"}),
            )

        try:
            pipeline = parse_dot_string(dot_source)
        except ParseError as exc:
            return (
                HTTPStatus.OK,
                json.dumps({
                    "valid": False,
                    "errors": [f"Parse error: {exc}"],
                    "warnings": [],
                }),
            )

        results = validate_pipeline(pipeline)
        errors = [
            str(e) for e in results if e.level == ValidationLevel.ERROR
        ]
        warnings = [
            str(e) for e in results if e.level == ValidationLevel.WARNING
        ]

        return (
            HTTPStatus.OK,
            json.dumps({
                "valid": len(errors) == 0,
                "pipeline_name": pipeline.name,
                "errors": errors,
                "warnings": warnings,
            }),
        )

    def _handle_status(self, run_id: str) -> tuple[HTTPStatus, str]:
        """Handle GET /pipelines/{id} — get pipeline status."""
        run = self._runs.get(run_id)
        if run is None:
            return (
                HTTPStatus.NOT_FOUND,
                json.dumps({"error": f"Pipeline run '{run_id}' not found"}),
            )
        return (HTTPStatus.OK, json.dumps(run.to_dict()))

    def _handle_context(self, run_id: str) -> tuple[HTTPStatus, str]:
        """Handle GET /pipelines/{id}/context — get pipeline context."""
        run = self._runs.get(run_id)
        if run is None:
            return (
                HTTPStatus.NOT_FOUND,
                json.dumps({"error": f"Pipeline run '{run_id}' not found"}),
            )

        if run.status == PipelineRunStatus.RUNNING:
            return (
                HTTPStatus.OK,
                json.dumps({
                    "id": run_id,
                    "status": run.status.value,
                    "context": {},
                    "message": "Pipeline is still running",
                }),
            )

        return (
            HTTPStatus.OK,
            json.dumps({
                "id": run_id,
                "status": run.status.value,
                "context": run.context,
            }),
        )

    def _handle_events(self, run_id: str) -> tuple[HTTPStatus, str]:
        """Handle GET /pipelines/{id}/events — get collected events."""
        run = self._runs.get(run_id)
        if run is None:
            return (
                HTTPStatus.NOT_FOUND,
                json.dumps({"error": f"Pipeline run '{run_id}' not found"}),
            )

        return (
            HTTPStatus.OK,
            json.dumps({
                "id": run_id,
                "events": run.events,
            }),
        )

    def _handle_cancel(self, run_id: str) -> tuple[HTTPStatus, str]:
        """Handle POST /pipelines/{id}/cancel — cancel a running pipeline."""
        run = self._runs.get(run_id)
        if run is None:
            return (
                HTTPStatus.NOT_FOUND,
                json.dumps({"error": f"Pipeline run '{run_id}' not found"}),
            )

        if run.status != PipelineRunStatus.RUNNING:
            return (
                HTTPStatus.CONFLICT,
                json.dumps({
                    "error": f"Pipeline is not running (status: {run.status.value})"
                }),
            )

        if run.task and not run.task.done():
            run.task.cancel()

        run.status = PipelineRunStatus.CANCELLED
        run.completed_at = time.time()

        return (
            HTTPStatus.OK,
            json.dumps({"id": run_id, "status": run.status.value}),
        )
