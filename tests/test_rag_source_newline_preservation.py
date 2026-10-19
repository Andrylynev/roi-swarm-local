"""Indexing must not normalize interior source newlines before chunking."""

import pytest

import roi_swarm.rag_cli as module
from roi_swarm.rag import HybridRagIndex


@pytest.mark.parametrize(
    "text",
    [
        "α\nβ\nγ",
        "α\r\nβ\r\nγ",
        "α\rβ\rγ",
        "α\r\nβ\nγ\rδ",
        "start\r\n" + "evidence\r\n" * 250 + "end",
    ],
    ids=["lf", "crlf", "cr", "mixed", "multiple-chunks"],
)
def test_indexed_chunks_preserve_decoded_source_newlines(tmp_path, monkeypatch, capsys, text):
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "guide.md"
    raw = text.encode("utf-8")
    source.write_bytes(raw)
    database = tmp_path / "index.sqlite"
    instances = []

    def local_index(path, embedder=None):
        assert embedder is None
        index = HybridRagIndex(database)
        instances.append(index)
        return index

    monkeypatch.setattr(module, "HybridRagIndex", local_index)
    module.index_directory(documents, collection="docs", embeddings=False)

    reopened = HybridRagIndex(database)
    try:
        rows = reopened.connection.execute(
            "SELECT collection_name, source, text FROM chunks"
        ).fetchall()
        actual = {row["source"]: (row["collection_name"], row["text"]) for row in rows}
        expected = {
            f"{source.as_posix()}#chunk-{number}": ("docs", chunk)
            for number, chunk in enumerate(module._chunks(text), start=1)
        }
        assert actual == expected
        assert len(instances) == 1
        assert source.read_bytes() == raw
        assert f"Indexed {len(expected)} chunks" in capsys.readouterr().out
    finally:
        reopened.close()
