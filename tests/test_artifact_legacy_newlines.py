"""Legacy newline recovery must be verified by the full content address."""

import hashlib
import json
from pathlib import Path

import pytest

from roi_swarm.artifacts import ArtifactStore


def _seed_artifact(store: ArtifactStore, text: str, raw: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    artifact_id = f"sha256-{digest}"
    path = store.content_dir / f"{artifact_id}.txt"
    path.write_bytes(raw)
    return artifact_id, path


@pytest.mark.parametrize(
    "raw",
    [
        b"unrelated\r\ncontent\r\n",
        b"first\r\r\nsecond\r\r\n",
        b"\xff\r\n",
    ],
    ids=["unrelated", "translated-twice", "invalid-utf8"],
)
def test_get_text_rejects_unrecoverable_content(tmp_path, raw):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id, path = _seed_artifact(store, "first\nsecond\n", raw)

    with pytest.raises(ValueError, match="Artifact content hash mismatch"):
        store.get_text(artifact_id)

    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "text",
    [
        "first\nsecond\n",
        "first\r\nsecond\r\n",
        "α\r\nβ\nγ\rδ\n\n",
        "first\r\r\nsecond\r\n",
    ],
    ids=["lf", "crlf", "mixed-unicode", "consecutive-cr"],
)
def test_get_text_recovers_one_windows_translation_without_writing(tmp_path, text):
    store = ArtifactStore(tmp_path / "artifacts")
    raw = text.encode("utf-8").replace(b"\n", b"\r\n")
    artifact_id, path = _seed_artifact(store, text, raw)
    reopened = ArtifactStore(store.root)

    restored = reopened.get_text(artifact_id)

    assert restored == text
    assert hashlib.sha256(restored.encode("utf-8")).hexdigest() == artifact_id[7:]
    assert path.read_bytes() == raw
    assert list(store.meta_dir.iterdir()) == []


@pytest.mark.parametrize("text", ["first\r\nsecond\r\n", "α\r\nβ\nγ\rδ\n"])
def test_get_text_preserves_valid_raw_hash_before_trying_recovery(tmp_path, text):
    store = ArtifactStore(tmp_path / "artifacts")
    raw = text.encode("utf-8")
    artifact_id, path = _seed_artifact(store, text, raw)

    assert store.get_text(artifact_id) == text
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"first\r\r\nsecond\r\r\n",
        b"first\nsecond\n",
        b"first",
        b"\xff\r\n",
    ],
    ids=["legacy-crlf", "other-valid-newlines", "partial-write", "invalid-utf8"],
)
def test_put_text_repairs_existing_bytes_from_known_original(tmp_path, raw):
    store = ArtifactStore(tmp_path / "artifacts")
    text = "first\r\nsecond\r\n"
    original = store.put_text("original-run", text, kind="tool-result")
    path = Path(original.path)
    path.write_bytes(raw)
    reopened = ArtifactStore(store.root)

    repaired = reopened.put_text("repair-run", text, kind="tool-result")

    assert repaired == original
    assert path.read_bytes() == text.encode("utf-8")
    assert reopened.get_text(original.id) == text
    metadata = json.loads((store.meta_dir / f"{original.id}.json").read_text(encoding="utf-8"))
    assert metadata["sha256"] == original.sha256
    assert metadata["run_ids"] == ["original-run", "repair-run"]


def test_put_text_does_not_rewrite_existing_correct_crlf(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    text = "first\r\nsecond\r\n"
    ref = store.put_text("same-run", text)
    original_write = ArtifactStore._atomic_write

    def reject_content_write(path, content):
        assert path.parent != store.content_dir, "Valid content must not be rewritten"
        return original_write(path, content)

    monkeypatch.setattr(ArtifactStore, "_atomic_write", staticmethod(reject_content_write))

    assert store.put_text("same-run", text) == ref
    assert Path(ref.path).read_bytes() == text.encode("utf-8")


def test_get_text_checks_entire_hash_in_identifier(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    text = "first\r\nsecond\r\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    wrong_last_digit = "0" if digest[-1] != "0" else "1"
    artifact_id = f"sha256-{digest[:-1]}{wrong_last_digit}"
    raw = text.encode("utf-8")
    path = store.content_dir / f"{artifact_id}.txt"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="Artifact content hash mismatch"):
        store.get_text(artifact_id)

    assert path.read_bytes() == raw
