from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from array import array
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class RagResult:
    id: str
    collection: str
    source: str
    text: str
    score: float
    lexical_score: float
    vector_score: float
    provenance: dict[str, str]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "collection": self.collection,
            "source": self.source,
            "text": self.text,
            "score": self.score,
            "lexical_score": self.lexical_score,
            "vector_score": self.vector_score,
            "provenance": self.provenance,
        }


class HybridRagIndex:
    def __init__(self, path: str | Path, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._setup()

    def upsert(self, *, collection: str, source: str, text: str) -> str:
        chunk_id = hashlib.sha256(f"{collection}\0{source}\0{text}".encode()).hexdigest()[:24]
        vector_blob = None
        if self.embedder is not None:
            vectors = self.embedder.embed([text])
            if vectors:
                vector_blob = array("f", vectors[0]).tobytes()
        with self.connection:
            # The ID binds collection, source and text. A lexical-only refresh of
            # that same chunk must not erase its previously computed embedding.
            self.connection.execute(
                """
                INSERT INTO chunks(id, collection_name, source, text, vector)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection_name=excluded.collection_name,
                    source=excluded.source,
                    text=excluded.text,
                    vector=COALESCE(excluded.vector, chunks.vector)
                """,
                (chunk_id, collection, source, text, vector_blob),
            )
            self.connection.execute("DELETE FROM chunks_fts WHERE id = ?", (chunk_id,))
            self.connection.execute(
                "INSERT INTO chunks_fts(id, text, source, collection_name) VALUES (?, ?, ?, ?)",
                (chunk_id, text, source, collection),
            )
        return chunk_id

    def search(
        self,
        query: str,
        *,
        collections: list[str] | None = None,
        top_k: int = 6,
        semantic: bool = True,
    ) -> list[RagResult]:
        """Retrieve up to top_k matches; zero skips SQL and embedding work."""
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if top_k == 0 or not query.strip() or collections == []:
            return []
        lexical = self._lexical_search(query, collections, top_k=max(top_k * 4, 20))
        vector = (
            self._vector_search(query, collections)
            if semantic and self.embedder is not None
            else {}
        )
        ids = set(lexical) | set(vector)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT id, collection_name, source, text FROM chunks WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        results: list[RagResult] = []
        for row in rows:
            lex = lexical.get(row["id"], 0.0)
            vec = vector.get(row["id"], 0.0)
            score = 0.55 * lex + 0.45 * vec if vector else lex
            results.append(
                RagResult(
                    id=row["id"],
                    collection=row["collection_name"],
                    source=row["source"],
                    text=row["text"],
                    score=score,
                    lexical_score=lex,
                    vector_score=vec,
                    provenance={
                        "collection": row["collection_name"],
                        "source": row["source"],
                        "chunk_id": row["id"],
                    },
                )
            )
        # SQL row order is unspecified; stable IDs resolve equal-score ties.
        results.sort(key=lambda item: (-item.score, item.id))
        return results[:top_k]

    def close(self) -> None:
        self.connection.close()

    def _setup(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks(
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vector BLOB
                )
                """
            )
            self.connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED,
                    text,
                    source,
                    collection_name UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )

    def _lexical_search(
        self,
        query: str,
        collections: list[str] | None,
        top_k: int,
    ) -> dict[str, float]:
        # Repeated request text must not create repeated FTS clauses or give
        # repeated words extra BM25 weight. Preserve first-occurrence order.
        terms = list(dict.fromkeys(
            term
            for term in re.findall(r"[\w-]+", query.lower(), flags=re.UNICODE)
            if len(term) >= 4
        ))
        if not terms:
            return {}
        expression = " OR ".join(f'"{term}"*' for term in terms)
        sql = "SELECT id, bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list = [expression]
        if collections:
            placeholders = ",".join("?" for _ in collections)
            sql += f" AND collection_name IN ({placeholders})"
            params.extend(collections)
        sql += " ORDER BY rank, id LIMIT ?"
        params.append(top_k)
        rows = self.connection.execute(sql, params).fetchall()
        if not rows:
            return {}
        raw_scores = [-float(row["rank"]) for row in rows]
        minimum = min(raw_scores)
        maximum = max(raw_scores)
        if maximum == minimum:
            return {row["id"]: 1.0 for row in rows}
        return {
            row["id"]: (score - minimum) / (maximum - minimum)
            for row, score in zip(rows, raw_scores, strict=True)
        }

    def _vector_search(self, query: str, collections: list[str] | None) -> dict[str, float]:
        sql = "SELECT id, vector FROM chunks WHERE vector IS NOT NULL"
        params: list[str] = []
        if collections:
            placeholders = ",".join("?" for _ in collections)
            sql += f" AND collection_name IN ({placeholders})"
            params.extend(collections)
        cursor = self.connection.execute(sql, params)
        try:
            # Avoid loading a lazy embedder when there is nothing in scope to score.
            # Retain the first row so this probe does not drop a candidate.
            first = cursor.fetchone()
            if first is None:
                return {}
            query_vectors = self.embedder.embed([query]) if self.embedder else []
            if not query_vectors:
                return {}
            query_vector = query_vectors[0]
            scores: dict[str, float] = {}
            for row in chain((first,), cursor):
                candidate = list(array("f", row["vector"]))
                scores[row["id"]] = max(0.0, self._cosine(query_vector, candidate))
            return scores
        finally:
            cursor.close()

    @staticmethod
    def _cosine(first: list[float], second: list[float]) -> float:
        if len(first) != len(second) or not first:
            return 0.0
        dot = sum(a * b for a, b in zip(first, second, strict=True))
        left = math.sqrt(sum(value * value for value in first))
        right = math.sqrt(sum(value * value for value in second))
        return dot / (left * right) if left and right else 0.0
