"""Unsupported document names should incur no per-file metadata probes."""

from pathlib import Path

import roi_swarm.rag_cli as module


def test_suffix_filter_precedes_file_probe(tmp_path, monkeypatch, capsys):
    root = tmp_path / "documents"
    root.mkdir()
    document = root / "guide.TXT"
    document.write_bytes(b"Exact\r\nevidence")
    unsupported = [root / name for name in ("image.png", "archive.bin", "LICENSE")]
    for file in unsupported:
        file.write_bytes(b"not a supported document")
    records = []
    closed = []
    probes = []
    original_is_file = Path.is_file

    def checked_is_file(file):
        assert file not in unsupported, "Unsupported suffix must skip metadata lookup"
        if file == document:
            probes.append(file)
        return original_is_file(file)

    class RecordingIndex:
        def __init__(self, path, embedder=None):
            assert embedder is None

        def upsert(self, **record):
            records.append(record)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "HybridRagIndex", RecordingIndex)
    monkeypatch.setattr(Path, "is_file", checked_is_file)

    module.index_directory(root, collection="docs", embeddings=False)

    assert probes == [document]
    assert records == [{
        "collection": "docs",
        "source": f"{document.as_posix()}#chunk-1",
        "text": "Exact\r\nevidence",
    }]
    assert closed == [True]
    assert "Indexed 1 chunks" in capsys.readouterr().out
