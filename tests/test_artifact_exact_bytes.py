"""Exact artifact content must agree with its UTF-8 content address."""

import hashlib
import json
from pathlib import Path

import pytest

from roi_swarm.artifacts import ArtifactStore


@pytest.mark.parametrize(
    "text",
    [
        "first\nsecond\n",
        "first\r\nsecond\r\n",
        "first\rsecond\r",
        "α\r\nβ\nγ\rδ\n\n",
        "",
    ],
)
def test_artifact_newlines_round_trip_without_changing_digest(tmp_path, text):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_text("newline-one", text)
    expected_bytes = text.encode("utf-8")

    assert Path(ref.path).read_bytes() == expected_bytes
    assert store.get_text(ref.id) == text
    assert ref.size_chars == len(text)
    assert ref.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert hashlib.sha256(store.get_text(ref.id).encode("utf-8")).hexdigest() == ref.sha256

    reopened = ArtifactStore(tmp_path / "artifacts")
    repeated = reopened.put_text("newline-two", text)
    assert repeated.id == ref.id
    assert reopened.get_text(ref.id) == text
    metadata = json.loads((store.meta_dir / f"{ref.id}.json").read_text(encoding="utf-8"))
    assert metadata["run_ids"] == ["newline-one", "newline-two"]


def test_artifacts_with_different_newlines_remain_distinct(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    texts = ["first\nsecond", "first\r\nsecond", "first\rsecond"]
    refs = [store.put_text("newline-variants", text) for text in texts]

    assert len({ref.id for ref in refs}) == len(texts)
    assert [store.get_text(ref.id) for ref in refs] == texts
