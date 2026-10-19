"""Python tool caches should not consume RAG traversal or supply evidence."""

from pathlib import Path

import pytest

import roi_swarm.rag_cli as module

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")


@pytest.mark.parametrize("root_name", ["documents", *CACHE_NAMES])
def test_python_caches_are_pruned_but_explicit_root_is_indexed(
    tmp_path, monkeypatch, capsys, root_name
):
    root = tmp_path / root_name
    chapter = root / "chapter"
    chapter.mkdir(parents=True)
    (root / "guide.md").write_text("Root evidence", encoding="utf-8")
    (chapter / "detail.json").write_text('{"evidence": true}', encoding="utf-8")
    for parent in (root, chapter):
        for name in CACHE_NAMES:
            cache = parent / name / "nested"
            cache.mkdir(parents=True)
            # These suffixes would otherwise qualify for indexing. Invalid bytes
            # also ensure that accidentally reading a cache cannot pass quietly.
            (cache / "cache.json").write_bytes(b"\xff")
            (cache / "README.md").write_text("Cache noise", encoding="utf-8")

    visited = []
    records = []
    closed = []
    original_walk = module.os.walk

    def recording_walk(*args, **kwargs):
        for folder, directories, filenames in original_walk(*args, **kwargs):
            visited.append(Path(folder))
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

    module.index_directory(root, collection="docs", embeddings=False)

    assert set(visited) == {root, chapter}
    assert sorted(records, key=lambda item: item["source"]) == sorted(
        [
            {
                "collection": "docs",
                "source": f"{(root / 'guide.md').as_posix()}#chunk-1",
                "text": "Root evidence",
            },
            {
                "collection": "docs",
                "source": f"{(chapter / 'detail.json').as_posix()}#chunk-1",
                "text": '{"evidence": true}',
            },
        ],
        key=lambda item: item["source"],
    )
    assert closed == [True]
    assert "Indexed 2 chunks" in capsys.readouterr().out
