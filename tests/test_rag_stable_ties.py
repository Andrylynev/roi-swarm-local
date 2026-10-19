"""Offline contracts for stable retrieval when relevance scores tie."""

import pytest

from roi_swarm.rag import HybridRagIndex


class ConstantEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize("semantic", [False, True])
def test_equal_score_results_are_stable_across_sql_row_orders(tmp_path, semantic):
    index = HybridRagIndex(
        tmp_path / "ties.sqlite",
        embedder=ConstantEmbedder() if semantic else None,
    )
    try:
        ids = [
            index.upsert(collection="docs", source=f"s{i}", text="matching evidence")
            for i in range(12)
        ]
        expected = sorted(ids)[:5]
        for reverse in (False, True):
            index.connection.execute(
                f"PRAGMA reverse_unordered_selects = {int(reverse)}"
            )
            results = index.search("matching", collections=["docs"], top_k=5)
            assert [item.id for item in results] == expected
            assert len({item.score for item in results}) == 1
    finally:
        index.close()


def test_lexical_candidate_limit_resolves_ties_before_truncation(tmp_path):
    index = HybridRagIndex(tmp_path / "candidate-ties.sqlite")
    try:
        ids = [
            index.upsert(collection="docs", source=f"s{i}", text="matching evidence")
            for i in range(40)
        ]
        expected = sorted(ids)[:3]
        first = index.search("matching", collections=["docs"], top_k=3)
        assert [item.id for item in first] == expected
        # Reinsert FTS rows in reverse order without changing their contents or IDs.
        for i in reversed(range(40)):
            index.upsert(collection="docs", source=f"s{i}", text="matching evidence")
        second = index.search("matching", collections=["docs"], top_k=3)
        assert [item.as_dict() for item in second] == [item.as_dict() for item in first]
    finally:
        index.close()


def test_relevance_score_still_precedes_chunk_id(tmp_path):
    class DistinctEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] if text == "preferred" else [0.0, 1.0] for text in texts]

    index = HybridRagIndex(tmp_path / "scores.sqlite", embedder=DistinctEmbedder())
    try:
        preferred = index.upsert(collection="docs", source="a", text="preferred")
        other = index.upsert(collection="docs", source="b", text="different")
        results = index.search("preferred", collections=["docs"])
        assert [item.id for item in results] == [preferred, other]
        assert results[0].score > results[1].score
    finally:
        index.close()


@pytest.mark.parametrize("reverse_unordered", [False, True])
def test_lexical_ties_survive_reverse_fts_insertion_and_sql_scan(tmp_path, reverse_unordered):
    index = HybridRagIndex(tmp_path / "reverse-candidate-ties.sqlite")
    try:
        chunks = [
            (
                index.upsert(collection="docs", source=f"s{i}", text="matching evidence"),
                f"s{i}",
            )
            for i in range(40)
        ]
        expected = sorted(chunk_id for chunk_id, _ in chunks)[:3]
        # Put the smallest IDs beyond the 20-row lexical candidate window.
        for chunk_id, source in sorted(chunks, reverse=True):
            assert index.upsert(
                collection="docs", source=source, text="matching evidence"
            ) == chunk_id
        index.connection.execute(f"PRAGMA reverse_unordered_selects = {int(reverse_unordered)}")

        results = index.search("matching", collections=["docs"], top_k=3, semantic=False)

        assert [item.id for item in results] == expected
        assert len({item.score for item in results}) == 1
    finally:
        index.close()
