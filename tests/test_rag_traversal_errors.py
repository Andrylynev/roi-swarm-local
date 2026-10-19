"""Traversal failures must not produce a successful partial-indexing summary."""

import pytest

import roi_swarm.rag_cli as module


@pytest.mark.parametrize("after_document", [False, True])
def test_scan_error_propagates_and_closes_index(
    tmp_path, monkeypatch, capsys, after_document
):
    root = tmp_path / "documents"
    root.mkdir()
    (root / "guide.md").write_text("Available evidence", encoding="utf-8")
    failure = PermissionError("fixture directory is unreadable")
    records = []
    closed = []

    class RecordingIndex:
        def __init__(self, path, embedder=None):
            assert embedder is None

        def upsert(self, **record):
            records.append(record)

        def close(self):
            closed.append(True)

    def failing_walk(path, topdown=True, onerror=None):
        assert path == root
        assert topdown is True
        if after_document:
            yield str(root), [], ["guide.md"]
        # Simulate os.walk's documented scan-error behavior without relying on
        # account permissions (privileged accounts can read chmod-zero folders).
        if onerror is not None:
            onerror(failure)

    monkeypatch.setattr(module, "HybridRagIndex", RecordingIndex)
    monkeypatch.setattr(module.os, "walk", failing_walk)

    with pytest.raises(PermissionError) as caught:
        module.index_directory(root, collection="docs", embeddings=False)

    assert caught.value is failure
    assert closed == [True]
    assert len(records) == int(after_document)
    if after_document:
        assert records[0]["text"] == "Available evidence"
    assert "Indexed" not in capsys.readouterr().out
