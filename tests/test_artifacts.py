from roi_swarm.artifacts import ArtifactStore


def test_artifact_store_deduplicates_by_sha256(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    first = store.put_text("run-a", "same content", kind="tool-result")
    second = store.put_text("run-a", "same content", kind="tool-result")

    assert first.id == second.id
    assert first.sha256 == second.sha256
    assert store.get_text(first.id) == "same content"


def test_artifact_store_rejects_invalid_identifiers(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    try:
        store.get_text("../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe artifact id was accepted")
