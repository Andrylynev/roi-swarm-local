"""Directory exclusions must not depend on ancestors of the selected input root."""

import pytest

import roi_swarm.rag_cli as module


@pytest.mark.parametrize("ancestor", ["data", "runs", "node_modules", ".venv", ".git"])
@pytest.mark.parametrize("select_ancestor", [False, True])
def test_index_root_ancestors_do_not_suppress_documents(
    tmp_path, monkeypatch, capsys, ancestor, select_ancestor
):
    parent = tmp_path / ancestor
    root = parent if select_ancestor else parent / "documents"
    root.mkdir(parents=True)
    document = root / "guide.md"
    document.write_text("Local retrieval evidence", encoding="utf-8")
    nested = root / "chapters" / "detail.TXT"
    nested.parent.mkdir()
    nested.write_text("More evidence", encoding="utf-8")
    for excluded in ("data", "runs", "node_modules", ".venv", ".git"):
        directory = root / excluded
        directory.mkdir()
        (directory / "ignored.md").write_text("Excluded descendant", encoding="utf-8")
    (root / "ignored.bin").write_bytes(b"not a document")
    records = []
    instances = []

    class RecordingIndex:
        def __init__(self, path, embedder=None):
            assert embedder is None
            self.closed = False
            instances.append(self)

        def upsert(self, **record):
            records.append(record)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "HybridRagIndex", RecordingIndex)
    module.index_directory(root, collection="selected", embeddings=False)

    assert sorted(records, key=lambda record: record["source"]) == sorted(
        [
            {
                "collection": "selected",
                "source": f"{document.as_posix()}#chunk-1",
                "text": "Local retrieval evidence",
            },
            {
                "collection": "selected",
                "source": f"{nested.as_posix()}#chunk-1",
                "text": "More evidence",
            },
        ],
        key=lambda record: record["source"],
    )
    assert len(instances) == 1
    assert instances[0].closed
    assert "Indexed 2 chunks" in capsys.readouterr().out
