"""Tests for the pipeline artifact store."""

import pytest

from attractor.pipeline.artifact_store import (
    ArtifactInfo,
    ArtifactStore,
    FILE_BACKING_THRESHOLD,
    LocalArtifactStore,
    _byte_size,
)


class TestByteSize:
    """Tests for the _byte_size helper."""

    def test_str_ascii(self) -> None:
        assert _byte_size("hello") == 5

    def test_str_unicode(self) -> None:
        # Multi-byte chars
        assert _byte_size("\u00e9") == 2

    def test_bytes(self) -> None:
        assert _byte_size(b"\x00\x01\x02") == 3

    def test_empty_string(self) -> None:
        assert _byte_size("") == 0

    def test_empty_bytes(self) -> None:
        assert _byte_size(b"") == 0


class TestLocalArtifactStoreInMemory:
    """Tests for LocalArtifactStore with in-memory storage."""

    def test_store_and_retrieve_string(self) -> None:
        store = LocalArtifactStore()
        info = store.store("a1", "my artifact", "hello world")
        assert info.id == "a1"
        assert info.name == "my artifact"
        assert info.size_bytes == len("hello world")
        assert info.is_file_backed is False
        assert store.retrieve("a1") == "hello world"

    def test_store_and_retrieve_bytes(self) -> None:
        store = LocalArtifactStore()
        data = b"\x00\x01\x02\x03"
        info = store.store("b1", "binary", data)
        assert info.size_bytes == 4
        assert info.is_file_backed is False
        assert store.retrieve("b1") == data

    def test_has_returns_true_for_existing(self) -> None:
        store = LocalArtifactStore()
        store.store("x", "X", "data")
        assert store.has("x") is True

    def test_has_returns_false_for_missing(self) -> None:
        store = LocalArtifactStore()
        assert store.has("missing") is False

    def test_list_returns_all_infos(self) -> None:
        store = LocalArtifactStore()
        store.store("a", "Alpha", "aaa")
        store.store("b", "Beta", "bbb")
        infos = store.list()
        ids = {i.id for i in infos}
        assert ids == {"a", "b"}
        assert all(isinstance(i, ArtifactInfo) for i in infos)

    def test_list_empty_store(self) -> None:
        store = LocalArtifactStore()
        assert store.list() == []

    def test_remove_existing_artifact(self) -> None:
        store = LocalArtifactStore()
        store.store("r1", "removable", "data")
        store.remove("r1")
        assert store.has("r1") is False

    def test_remove_missing_raises_keyerror(self) -> None:
        store = LocalArtifactStore()
        with pytest.raises(KeyError, match="Artifact not found"):
            store.remove("nonexistent")

    def test_retrieve_missing_raises_keyerror(self) -> None:
        store = LocalArtifactStore()
        with pytest.raises(KeyError, match="Artifact not found"):
            store.retrieve("nonexistent")

    def test_clear_removes_all(self) -> None:
        store = LocalArtifactStore()
        store.store("a", "A", "1")
        store.store("b", "B", "2")
        store.clear()
        assert store.list() == []
        assert store.has("a") is False
        assert store.has("b") is False

    def test_overwrite_existing_artifact(self) -> None:
        store = LocalArtifactStore()
        store.store("k", "v1", "old")
        store.store("k", "v2", "new")
        assert store.retrieve("k") == "new"
        info_list = store.list()
        assert len(info_list) == 1
        assert info_list[0].name == "v2"


class TestLocalArtifactStoreFileBacked:
    """Tests for LocalArtifactStore with file-backing threshold."""

    def test_small_artifact_stays_in_memory(self, tmp_path: ...) -> None:
        store = LocalArtifactStore(base_dir=tmp_path, threshold=100)
        info = store.store("small", "small data", "x" * 50)
        assert info.is_file_backed is False
        assert store.retrieve("small") == "x" * 50

    def test_large_artifact_is_file_backed(self, tmp_path: ...) -> None:
        store = LocalArtifactStore(base_dir=tmp_path, threshold=100)
        large_data = "x" * 200
        info = store.store("big", "big data", large_data)
        assert info.is_file_backed is True
        assert info.size_bytes == 200
        # File should exist on disk
        artifact_file = tmp_path / "artifacts" / "big.json"
        assert artifact_file.exists()
        assert artifact_file.read_text() == large_data
        # Retrieve should return the same data
        assert store.retrieve("big") == large_data

    def test_large_bytes_artifact_is_file_backed(self, tmp_path: ...) -> None:
        store = LocalArtifactStore(base_dir=tmp_path, threshold=100)
        large_data = b"\xff" * 200
        info = store.store("bigbytes", "big bytes", large_data)
        assert info.is_file_backed is True
        assert store.retrieve("bigbytes") == large_data

    def test_remove_cleans_up_file(self, tmp_path: ...) -> None:
        store = LocalArtifactStore(base_dir=tmp_path, threshold=100)
        store.store("file_rm", "removable", "x" * 200)
        artifact_file = tmp_path / "artifacts" / "file_rm.json"
        assert artifact_file.exists()
        store.remove("file_rm")
        assert not artifact_file.exists()

    def test_clear_cleans_up_files(self, tmp_path: ...) -> None:
        store = LocalArtifactStore(base_dir=tmp_path, threshold=100)
        store.store("f1", "file1", "x" * 200)
        store.store("f2", "file2", "y" * 200)
        store.clear()
        assert not (tmp_path / "artifacts" / "f1.json").exists()
        assert not (tmp_path / "artifacts" / "f2.json").exists()

    def test_no_file_backing_without_base_dir(self) -> None:
        """Large artifacts stay in memory when base_dir is None."""
        store = LocalArtifactStore(base_dir=None, threshold=100)
        large_data = "x" * 200
        info = store.store("mem_only", "no disk", large_data)
        assert info.is_file_backed is False
        assert store.retrieve("mem_only") == large_data

    def test_default_threshold_is_100kb(self) -> None:
        assert FILE_BACKING_THRESHOLD == 100 * 1024


class TestArtifactStoreProtocol:
    """Verify LocalArtifactStore satisfies the ArtifactStore protocol."""

    def test_isinstance_check(self) -> None:
        store = LocalArtifactStore()
        assert isinstance(store, ArtifactStore)
