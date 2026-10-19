from roi_swarm.rag import HybridRagIndex


def test_hybrid_rag_indexes_and_retrieves_russian_text(tmp_path) -> None:
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=None)
    index.upsert(
        collection="docs",
        source="risk.md",
        text=(
            "Ликвидация фьючерсной позиции зависит от кредитного плеча и маржи; "
            "риск растёт с плечом."
        ),
    )
    index.upsert(
        collection="docs",
        source="booking.md",
        text="Система бронирования управляет календарём клиентов.",
    )

    results = index.search("риск ликвидации с плечом", collections=["docs"], top_k=2)

    assert results[0].source == "risk.md"
    assert results[0].text.startswith("Ликвидация")
    assert results[0].provenance["collection"] == "docs"


def test_rag_is_optional_when_no_index_matches(tmp_path) -> None:
    index = HybridRagIndex(tmp_path / "rag.sqlite", embedder=None)

    assert index.search("anything", top_k=3) == []
