"""Tests for PipelineContext methods added for S1-ctx and S4-ckpt."""

import threading

from attractor.pipeline.models import Checkpoint, PipelineContext


class TestGetString:
    def test_returns_string_value(self) -> None:
        ctx = PipelineContext()
        ctx.set("name", "alice")
        assert ctx.get_string("name") == "alice"

    def test_coerces_int_to_string(self) -> None:
        ctx = PipelineContext()
        ctx.set("count", 42)
        assert ctx.get_string("count") == "42"

    def test_coerces_bool_to_string(self) -> None:
        ctx = PipelineContext()
        ctx.set("flag", True)
        assert ctx.get_string("flag") == "True"

    def test_coerces_float_to_string(self) -> None:
        ctx = PipelineContext()
        ctx.set("ratio", 3.14)
        assert ctx.get_string("ratio") == "3.14"

    def test_missing_key_returns_default(self) -> None:
        ctx = PipelineContext()
        assert ctx.get_string("missing") == ""

    def test_missing_key_custom_default(self) -> None:
        ctx = PipelineContext()
        assert ctx.get_string("missing", "fallback") == "fallback"

    def test_none_value_returns_default(self) -> None:
        ctx = PipelineContext()
        ctx.set("key", None)
        assert ctx.get_string("key", "fallback") == "fallback"


class TestSnapshot:
    def test_returns_all_data(self) -> None:
        ctx = PipelineContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        snap = ctx.snapshot()
        assert snap == {"a": 1, "b": "two"}

    def test_deep_copy_isolation(self) -> None:
        ctx = PipelineContext()
        ctx.set("nested", {"inner": [1, 2, 3]})
        snap = ctx.snapshot()
        # Mutate the snapshot — should not affect the context
        snap["nested"]["inner"].append(4)
        assert ctx.get("nested") == {"inner": [1, 2, 3]}

    def test_shallow_to_dict_not_isolated(self) -> None:
        """Confirm that to_dict is shallow, so snapshot is needed for isolation."""
        ctx = PipelineContext()
        ctx.set("nested", {"inner": [1, 2, 3]})
        shallow = ctx.to_dict()
        # Mutate the shallow copy — DOES affect context (shared reference)
        shallow["nested"]["inner"].append(4)
        assert ctx.get("nested") == {"inner": [1, 2, 3, 4]}

    def test_empty_context(self) -> None:
        ctx = PipelineContext()
        assert ctx.snapshot() == {}


class TestAppendLog:
    def test_append_single_entry(self) -> None:
        ctx = PipelineContext()
        entry = {"event": "start", "node": "a"}
        ctx.append_log(entry)
        logs = ctx.get_logs()
        assert len(logs) == 1
        assert logs[0] == {"event": "start", "node": "a"}

    def test_append_multiple_entries(self) -> None:
        ctx = PipelineContext()
        ctx.append_log({"event": "start"})
        ctx.append_log({"event": "complete"})
        ctx.append_log({"event": "end"})
        logs = ctx.get_logs()
        assert len(logs) == 3
        assert logs[0]["event"] == "start"
        assert logs[2]["event"] == "end"

    def test_preserves_order(self) -> None:
        ctx = PipelineContext()
        for i in range(5):
            ctx.append_log({"seq": i})
        logs = ctx.get_logs()
        assert [entry["seq"] for entry in logs] == [0, 1, 2, 3, 4]


class TestGetLogs:
    def test_empty_initially(self) -> None:
        ctx = PipelineContext()
        assert ctx.get_logs() == []

    def test_returns_copy(self) -> None:
        ctx = PipelineContext()
        ctx.append_log({"event": "test"})
        logs = ctx.get_logs()
        logs.append({"event": "injected"})
        assert len(ctx.get_logs()) == 1


class TestCloneWithLogs:
    def test_clone_copies_logs(self) -> None:
        ctx = PipelineContext()
        ctx.set("key", "value")
        ctx.append_log({"event": "original"})
        cloned = ctx.clone()
        assert cloned.get_logs() == [{"event": "original"}]
        assert cloned.get("key") == "value"

    def test_clone_logs_isolated(self) -> None:
        ctx = PipelineContext()
        ctx.append_log({"event": "original"})
        cloned = ctx.clone()
        cloned.append_log({"event": "cloned_only"})
        assert len(ctx.get_logs()) == 1
        assert len(cloned.get_logs()) == 2


class TestThreadSafety:
    def test_concurrent_set_get(self) -> None:
        ctx = PipelineContext()
        def writer(start: int) -> None:
            for i in range(100):
                ctx.set(f"key_{start}_{i}", i)

        def reader() -> None:
            for _ in range(100):
                ctx.get("key_0_0")

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all writes landed
        for i in range(100):
            assert ctx.get(f"key_0_{i}") == i
            assert ctx.get(f"key_1_{i}") == i

    def test_concurrent_append_log(self) -> None:
        ctx = PipelineContext()

        def appender(thread_id: int) -> None:
            for i in range(50):
                ctx.append_log({"thread": thread_id, "seq": i})

        threads = [threading.Thread(target=appender, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ctx.get_logs()) == 200

    def test_concurrent_update_and_snapshot(self) -> None:
        ctx = PipelineContext()
        ctx.set("counter", 0)

        def updater() -> None:
            for i in range(100):
                ctx.update({"counter": i})

        def snapshotter() -> None:
            for _ in range(50):
                snap = ctx.snapshot()
                assert "counter" in snap
                assert isinstance(snap["counter"], int)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=snapshotter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestCheckpointLogs:
    """S4-ckpt: Verify that PipelineContext logs can be captured in Checkpoint."""

    def test_checkpoint_captures_context_logs(self) -> None:
        ctx = PipelineContext()
        ctx.set("status", "running")
        ctx.append_log({"event": "node_started", "node": "a"})
        ctx.append_log({"event": "node_completed", "node": "a"})

        checkpoint = Checkpoint(
            pipeline_name="test",
            current_node="b",
            context=ctx,
            completed_nodes=["a"],
            logs=ctx.get_logs(),
        )
        assert len(checkpoint.logs) == 2
        assert checkpoint.logs[0]["event"] == "node_started"

    def test_checkpoint_roundtrip_preserves_logs(self) -> None:
        ctx = PipelineContext()
        ctx.append_log({"event": "start", "ts": 1234})

        checkpoint = Checkpoint(
            pipeline_name="test",
            current_node="a",
            context=ctx,
            logs=ctx.get_logs(),
        )
        data = checkpoint.to_dict()
        restored = Checkpoint.from_dict(data)
        assert restored.logs == [{"event": "start", "ts": 1234}]

    def test_checkpoint_file_roundtrip_with_logs(self, tmp_path: object) -> None:
        import pathlib

        tmp = pathlib.Path(str(tmp_path))
        ctx = PipelineContext()
        ctx.append_log({"event": "saved"})

        checkpoint = Checkpoint(
            pipeline_name="test",
            current_node="node1",
            context=ctx,
            logs=ctx.get_logs(),
        )
        filepath = tmp / "ckpt.json"
        checkpoint.save_to_file(filepath)
        restored = Checkpoint.load_from_file(filepath)
        assert restored.logs == [{"event": "saved"}]
