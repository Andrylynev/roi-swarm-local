"""Local contracts for avoiding embeddings without scoped vector candidates."""

import pytest

from roi_swarm.rag import HybridRagIndex


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize("collections", [None, ["docs"], ["missing"]])
def test_empty_index_does_not_embed_query(tmp_path, collections):
    embedder = RecordingEmbedder()
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=embedder)
    try:
        assert index.search("retrieval", collections=collections) == []
        assert embedder.calls == []
    finally:
        index.close()


def test_foreign_vectors_do_not_trigger_embedding_for_lexical_only_scope(tmp_path):
    index = HybridRagIndex(tmp_path / "rag.sqlite")
    embedder = RecordingEmbedder()
    try:
        index.upsert(collection="docs", source="local", text="retrieval evidence")
        expected = index.search("retrieval", collections=["docs"], semantic=False)
        index.embedder = embedder
        index.upsert(collection="foreign", source="other", text="retrieval evidence")
        embedder.calls.clear()

        actual = index.search("retrieval", collections=["docs"])

        assert len(actual) == 1
        assert [item.as_dict() for item in actual] == [item.as_dict() for item in expected]
        assert index.search("retrieval", collections=["missing"]) == []
        assert embedder.calls == []
    finally:
        index.close()


@pytest.mark.parametrize("count", [1, 3])
def test_candidate_probe_preserves_every_vector_and_embeds_query_once(tmp_path, count):
    embedder = RecordingEmbedder()
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=embedder)
    try:
        ids = {
            index.upsert(collection="docs", source=str(number), text="candidate")
            for number in range(count)
        }
        embedder.calls.clear()

        # No lexical overlap: every result must come from the vector scan.
        results = index.search("unrelated", collections=["docs"])

        assert {item.id for item in results} == ids
        assert all(item.vector_score == 1.0 for item in results)
        assert all(item.lexical_score == 0.0 for item in results)
        assert embedder.calls == [["unrelated"]]
    finally:
        index.close()
