"""Offline retrieval-limit contracts using a local index and recording embedder."""

import pytest

from roi_swarm.rag import HybridRagIndex


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize("semantic", [False, True])
def test_zero_result_limit_skips_sql_and_embeddings(tmp_path, semantic):
    embedder = RecordingEmbedder()
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=embedder)
    try:
        index.upsert(collection="docs", source="one", text="retrieval evidence")
        embedder.calls.clear()
        statements = []
        index.connection.set_trace_callback(statements.append)

        assert index.search("retrieval", top_k=0, semantic=semantic) == []
        assert statements == []
        assert embedder.calls == []
    finally:
        index.close()


@pytest.mark.parametrize("limit", [-1, -10, True, False, 1.5, "2", None])
@pytest.mark.parametrize("query,collections", [("retrieval", None), ("", [])])
def test_invalid_result_limits_fail_before_retrieval(tmp_path, limit, query, collections):
    embedder = RecordingEmbedder()
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=embedder)
    try:
        statements = []
        index.connection.set_trace_callback(statements.append)
        with pytest.raises(ValueError, match="top_k must be a nonnegative integer"):
            index.search(query, collections=collections, top_k=limit)
        assert statements == []
        assert embedder.calls == []
    finally:
        index.close()


@pytest.mark.parametrize("semantic", [False, True])
def test_positive_result_limits_preserve_retrieval_and_collection_scope(tmp_path, semantic):
    embedder = RecordingEmbedder()
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=embedder)
    try:
        for source in ("one", "two", "three"):
            index.upsert(collection="docs", source=source, text="retrieval evidence")
        index.upsert(collection="other", source="foreign", text="retrieval evidence")
        embedder.calls.clear()

        results = index.search("retrieval", collections=["docs"], top_k=2, semantic=semantic)

        assert len(results) == 2
        assert all(result.collection == "docs" for result in results)
        assert all(result.source != "foreign" for result in results)
        assert embedder.calls == ([["retrieval"]] if semantic else [])
    finally:
        index.close()
