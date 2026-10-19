"""Excluded indexing trees should not be traversed, not merely left unindexed."""

from pathlib import Path

import pytest

import roi_swarm.rag_cli as module


@pytest.mark.parametrize("root_name", ["documents", "data", "node_modules"])
def test_indexing_prunes_excluded_descendants_before_descent(
    tmp_path, monkeypatch, capsys, root_name
):
    root = tmp_path / root_name
    root.mkdir()
    (root / "guide.md").write_text("Root evidence", encoding="utf-8")
    chapter = root / "chapters"
    chapter.mkdir()
    (chapter / "detail.TXT").write_text("Nested evidence", encoding="utf-8")
    (chapter / "ignored.bin").write_bytes(b"not indexed")
    excluded = {".git", ".venv", "node_modules", "runs", "data"}
    for parent in (root, chapter):
        for name in excluded:
            nested = parent / name / "deep"
            nested.mkdir(parents=True)
            (nested / "ignored.md").write_text("Excluded evidence", encoding="utf-8")

    visited = []
    records = []
    closed = []
    original_walk = module.os.walk

    def recording_walk(*args, **kwargs):
        for folder, directories, filenames in original_walk(*args, **kwargs):
            visited.append(Path(folder))
            # Forward the actual mutable list so pruning controls real descent.
            yield folder, directories, filenames

    class RecordingIndex:
        def __init__(self, path, embedder=None):
            assert embedder is None

        def upsert(self, **record):
            records.append(record)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module.os, "walk", recording_walk)
    monkeypatch.setattr(module, "HybridRagIndex", RecordingIndex)

    module.index_directory(root, collection="selected", embeddings=False)

    assert set(visited) == {root, chapter}
    assert sorted(records, key=lambda record: record["source"]) == sorted(
        [
            {
                "collection": "selected",
                "source": f"{(root / 'guide.md').as_posix()}#chunk-1",
                "text": "Root evidence",
            },
            {
                "collection": "selected",
                "source": f"{(chapter / 'detail.TXT').as_posix()}#chunk-1",
                "text": "Nested evidence",
            },
        ],
        key=lambda record: record["source"],
    )
    assert closed == [True]
    assert "Indexed 2 chunks" in capsys.readouterr().out
