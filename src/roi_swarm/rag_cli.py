from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from roi_swarm.embeddings import GraniteOnnxEmbedder
from roi_swarm.rag import HybridRagIndex

app = typer.Typer(no_args_is_help=True)
PROJECT_ROOT = Path(__file__).parents[2]


@app.command("index")
def index_directory(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    collection: Annotated[str, typer.Option()] = "docs",
    # Vector search needs the optional local ONNX model (`uv sync --extra embeddings`).
    embeddings: Annotated[bool, typer.Option()] = False,
) -> None:
    """Index markdown, text, Python, JSON and YAML files into local hybrid RAG."""
    embedder = (
        GraniteOnnxEmbedder(cache_dir=str(PROJECT_ROOT / "data" / "models" / "huggingface"))
        if embeddings
        else None
    )
    index = HybridRagIndex(PROJECT_ROOT / "data" / "rag" / "rag.sqlite", embedder=embedder)
    suffixes = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
    # Tool caches can contain supported JSON/text files as well as many irrelevant
    # entries. Prune them before descent, just like dependency and runtime trees.
    excluded = {
        ".git", ".venv", "node_modules", "runs", "data",
        "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    }
    count = 0

    def traversal_error(error: OSError) -> None:
        # os.walk otherwise ignores scan errors and can report partial indexing
        # as success. Existing upserts remain, but the caller must see the failure.
        raise error

    try:
        # Prune before descent rather than filtering after rglob has traversed
        # dependency trees and runtime data. The selected root itself is allowed,
        # even when its name (or an ancestor's name) is in the exclusion set.
        for folder, directories, filenames in os.walk(
            path, topdown=True, onerror=traversal_error
        ):
            directories[:] = [name for name in directories if name not in excluded]
            for filename in filenames:
                file = Path(folder) / filename
                # Reject unsupported names before probing filesystem metadata.
                if file.suffix.lower() not in suffixes or not file.is_file():
                    continue
                # Strict decoding preserves evidence, including interior CRLF/CR
                # sequences that text-mode universal newline handling would alter.
                text = file.read_bytes().decode("utf-8")
                for chunk_number, chunk in enumerate(_chunks(text), start=1):
                    source = f"{file.as_posix()}#chunk-{chunk_number}"
                    index.upsert(collection=collection, source=source, text=chunk)
                    count += 1
    finally:
        index.close()
    typer.echo(f"Indexed {count} chunks into collection '{collection}'")


def _chunks(text: str, chunk_size: int = 1800, overlap: int = 200):
    """Yield bounded overlapping chunks; reject settings that cannot advance normally."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer between zero and chunk_size - 1")
    cleaned = text.strip()
    if not cleaned:
        return
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        yield cleaned[start:end]
        if end == len(cleaned):
            break
        start = max(start + 1, end - overlap)


if __name__ == "__main__":
    app()
