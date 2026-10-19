"""Indexing must not silently remove undecodable bytes from source evidence."""

import pytest

import roi_swarm.rag_cli as module


@pytest.mark.parametrize("raw", [b"before\xffafter", b"truncated\xe2\x82"])
@pytest.mark.parametrize("with_prior_document", [False, True])
def test_invalid_utf8_propagates_closes_index_and_has_no_success_summary(
    tmp_path, monkeypatch, capsys, raw, with_prior_document
):
    root = tmp_path / "documents"
    root.mkdir()
    damaged = root / "damaged.py"
    damaged.write_bytes(raw)
    healthy = root / "guide.md"
    healthy.write_text("Проверенные данные", encoding="utf-8")
    records = []
    closed = []

    class RecordingIndex:
        def __init__(self, path, embedder=None):
            assert embedder is None

        def upsert(self, **record):
            records.append(record)

        def close(self):
            closed.append(True)

    def ordered_walk(path, **kwargs):
        assert path == root
        filenames = [healthy.name, damaged.name] if with_prior_document else [damaged.name]
        yield str(root), [], filenames

    monkeypatch.setattr(module, "HybridRagIndex", RecordingIndex)
    monkeypatch.setattr(module.os, "walk", ordered_walk)

    with pytest.raises(UnicodeDecodeError):
        module.index_directory(root, collection="docs", embeddings=False)

    assert closed == [True]
    assert records == (
        [{
            "collection": "docs",
            "source": f"{healthy.as_posix()}#chunk-1",
            "text": "Проверенные данные",
        }]
        if with_prior_document else []
    )
    assert damaged.read_bytes() == raw
    assert "Indexed" not in capsys.readouterr().out
