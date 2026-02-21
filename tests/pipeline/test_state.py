"""Tests for PipelineContext and Checkpoint serialization."""

import json
import tempfile
from pathlib import Path

from attractor.pipeline.models import Checkpoint, PipelineContext
from attractor.pipeline.state import latest_checkpoint, list_checkpoints, save_checkpoint


class TestPipelineContext:
    def test_get_set_has(self) -> None:
        ctx = PipelineContext()
        assert ctx.has("key") is False
        ctx.set("key", "value")
        assert ctx.has("key") is True
        assert ctx.get("key") == "value"

    def test_get_default(self) -> None:
        ctx = PipelineContext()
        assert ctx.get("missing", "default") == "default"
        assert ctx.get("missing") is None

    def test_delete(self) -> None:
        ctx = PipelineContext(data={"a": 1, "b": 2})
        ctx.delete("a")
        assert ctx.has("a") is False
        assert ctx.has("b") is True
        # Deleting non-existent key should not raise
        ctx.delete("nonexistent")

    def test_update(self) -> None:
        ctx = PipelineContext(data={"a": 1})
        ctx.update({"b": 2, "c": 3})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
        assert ctx.get("c") == 3

    def test_to_dict_from_dict(self) -> None:
        original = PipelineContext(data={"x": 42, "y": "hello"})
        serialized = original.to_dict()
        restored = PipelineContext.from_dict(serialized)
        assert restored.get("x") == 42
        assert restored.get("y") == "hello"

    def test_from_dict_is_independent_copy(self) -> None:
        data = {"a": 1}
        ctx = PipelineContext.from_dict(data)
        ctx.set("a", 999)
        assert data["a"] == 1  # original unchanged

    def test_scoped_context(self) -> None:
        parent = PipelineContext(data={"shared": True})
        scope = parent.create_scope("branch1")
        scope.set("result", "ok")
        parent.merge_scope(scope, "branch1")
        assert parent.get("branch1.result") == "ok"
        assert parent.get("shared") is True


class TestCheckpoint:
    def test_to_dict_from_dict(self) -> None:
        cp = Checkpoint(
            pipeline_name="test_pipeline",
            current_node="step2",
            context=PipelineContext(data={"key": "value"}),
            completed_nodes=["step1"],
            timestamp=1234567890.0,
        )
        data = cp.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.pipeline_name == "test_pipeline"
        assert restored.current_node == "step2"
        assert restored.context.get("key") == "value"
        assert restored.completed_nodes == ["step1"]
        assert restored.timestamp == 1234567890.0

    def test_save_and_load_file(self, tmp_path: Path) -> None:
        cp = Checkpoint(
            pipeline_name="file_test",
            current_node="node_a",
            context=PipelineContext(data={"count": 5}),
            completed_nodes=["init"],
        )
        path = tmp_path / "cp.json"
        cp.save_to_file(path)
        loaded = Checkpoint.load_from_file(path)
        assert loaded.pipeline_name == "file_test"
        assert loaded.context.get("count") == 5

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "cp.json"
        cp = Checkpoint(
            pipeline_name="deep",
            current_node="x",
            context=PipelineContext(),
        )
        cp.save_to_file(deep_path)
        assert deep_path.exists()


class TestStateHelpers:
    def test_save_checkpoint_and_list(self, tmp_path: Path) -> None:
        cp1 = Checkpoint(
            pipeline_name="p",
            current_node="a",
            context=PipelineContext(),
            timestamp=1000.0,
        )
        cp2 = Checkpoint(
            pipeline_name="p",
            current_node="b",
            context=PipelineContext(),
            timestamp=2000.0,
        )
        save_checkpoint(cp1, tmp_path)
        save_checkpoint(cp2, tmp_path)

        files = list_checkpoints(tmp_path)
        assert len(files) == 2
        # Newest first
        assert "2000000" in files[0].name

    def test_latest_checkpoint(self, tmp_path: Path) -> None:
        cp = Checkpoint(
            pipeline_name="latest_test",
            current_node="z",
            context=PipelineContext(data={"v": 1}),
            timestamp=9999.0,
        )
        save_checkpoint(cp, tmp_path)
        latest = latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.pipeline_name == "latest_test"

    def test_latest_checkpoint_empty_dir(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path) is None

    def test_list_checkpoints_nonexistent_dir(self) -> None:
        assert list_checkpoints("/nonexistent/path") == []
