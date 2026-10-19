"""Offline contracts for embeddings across lexical-only chunk refreshes."""

from array import array

from roi_swarm.rag import HybridRagIndex


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


def stored_vector(index, chunk_id):
    return index.connection.execute(
        "SELECT vector FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()[0]


def test_lexical_refresh_preserves_embedding_across_reopen(tmp_path):
    path = tmp_path / "rag.sqlite"
    embedder = FixedEmbedder([1.0, 0.0])
    chunk = dict(collection="docs", source="guide", text="retrieval evidence")
    index = HybridRagIndex(path, embedder=embedder)
    try:
        chunk_id = index.upsert(**chunk)
        original = stored_vector(index, chunk_id)
    finally:
        index.close()

    index = HybridRagIndex(path)
    try:
        for _ in range(3):
            assert index.upsert(**chunk) == chunk_id
        assert stored_vector(index, chunk_id) == original
        assert index.connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE id = ?", (chunk_id,)
        ).fetchone()[0] == 1
        assert [item.id for item in index.search("retrieval", semantic=False)] == [chunk_id]
        index.embedder = embedder
        embedder.calls.clear()
        results = index.search("unrelated", collections=["docs"])
        assert [item.id for item in results] == [chunk_id]
        assert results[0].vector_score == 1.0
        assert results[0].lexical_score == 0.0
        assert embedder.calls == [["unrelated"]]
        assert index.search("unrelated", collections=["other"]) == []
    finally:
        index.close()


def test_new_embeddings_can_populate_and_replace_existing_vectors(tmp_path):
    index = HybridRagIndex(tmp_path / "rag.sqlite")
    chunk = dict(collection="docs", source="guide", text="retrieval evidence")
    try:
        chunk_id = index.upsert(**chunk)
        assert stored_vector(index, chunk_id) is None
        index.embedder = FixedEmbedder([1.0, 0.0])
        assert index.upsert(**chunk) == chunk_id
        assert stored_vector(index, chunk_id) == array("f", [1.0, 0.0]).tobytes()
        index.embedder = FixedEmbedder([0.0, 1.0])
        assert index.upsert(**chunk) == chunk_id
        assert stored_vector(index, chunk_id) == array("f", [0.0, 1.0]).tobytes()

        index.embedder = None
        changed_id = index.upsert(**(chunk | {"text": "different evidence"}))
        assert changed_id != chunk_id
        assert stored_vector(index, changed_id) is None
        assert stored_vector(index, chunk_id) == array("f", [0.0, 1.0]).tobytes()
    finally:
        index.close()
