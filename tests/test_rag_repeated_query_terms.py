"""Repeated request words should not inflate lexical clauses or relevance weight."""

import pytest

from roi_swarm.rag import HybridRagIndex


@pytest.mark.parametrize("term,other", [("retrieval", "evidence"), ("проверка", "данные")])
def test_repeated_query_terms_match_unique_query_scores_and_scope(tmp_path, term, other):
    index = HybridRagIndex(tmp_path / "rag.sqlite")
    try:
        for source, text in (
            ("first", f"{term} {term}"),
            ("second", f"{other} {other}"),
            ("both", f"{term} {other}"),
        ):
            index.upsert(collection="docs", source=source, text=text)
        index.upsert(collection="foreign", source="excluded", text=f"{term} {other}")
        expected = index.search(
            f"{term} {other}", collections=["docs"], semantic=False
        )
        statements = []
        index.connection.set_trace_callback(statements.append)
        actual = index.search(
            f"{term.upper()} " * 1500 + other,
            collections=["docs"],
            semantic=False,
        )

        assert len(expected) == 3
        assert [item.as_dict() for item in actual] == [item.as_dict() for item in expected]
        assert all(item.collection == "docs" for item in actual)
        match_statements = [sql for sql in statements if "chunks_fts MATCH" in sql]
        assert len(match_statements) == 1
        assert match_statements[0].count(f'"{term}"*') == 1
        assert match_statements[0].count(f'"{other}"*') == 1
    finally:
        index.close()
