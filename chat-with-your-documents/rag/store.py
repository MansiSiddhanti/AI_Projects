from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .models import Chunk, DocumentInfo


class EmbeddingModelMismatch(RuntimeError):
    pass


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        path: str | Path | None = None,
        client: chromadb.api.ClientAPI | None = None,
    ):
        if client is None:
            client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        self.client = client
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedding_model:
            raise EmbeddingModelMismatch(
                f"Collection '{collection_name}' was built with '{stored_model}', "
                f"but the configured embedding model is '{embedding_model}'. "
                "Use a different RAG_COLLECTION_NAME or re-create the index."
            )
        # bumped on writes so the bm25 index knows to rebuild
        self.version = 0

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> list[Chunk]:
        data = self.collection.get(include=["documents", "metadatas"])
        return _to_chunks(data["ids"], data["documents"], data["metadatas"])

    def source_chunks(self, source: str) -> list[Chunk]:
        data = self.collection.get(where={"source": source}, include=["documents", "metadatas"])
        chunks = _to_chunks(data["ids"], data["documents"], data["metadatas"])
        return sorted(chunks, key=lambda c: (c.page, c.chunk_index))

    def source_hash(self, source: str) -> str | None:
        data = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        metadatas = data.get("metadatas") or []
        return (metadatas[0] or {}).get("doc_hash") if metadatas else None

    def documents(self) -> list[DocumentInfo]:
        data = self.collection.get(include=["metadatas"])
        pages: dict[str, set[int]] = defaultdict(set)
        chunks: dict[str, int] = defaultdict(int)
        hashes: dict[str, str] = {}
        for meta in data.get("metadatas") or []:
            if not meta or not meta.get("source"):
                continue
            source = meta["source"]
            pages[source].add(int(meta.get("page", 0)))
            chunks[source] += 1
            hashes[source] = meta.get("doc_hash", "")
        return [
            DocumentInfo(source=s, pages=len(pages[s]), chunks=chunks[s], doc_hash=hashes[s])
            for s in sorted(chunks, key=str.lower)
        ]

    def query(
        self, embedding: list[float], n: int, sources: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        total = self.count()
        if total == 0:
            return []
        where = {"source": {"$in": list(sources)}} if sources else None
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = _to_chunks(result["ids"][0], result["documents"][0], result["metadatas"][0])
        # chroma returns cosine distance
        return [(c, 1.0 - float(d)) for c, d in zip(chunks, result["distances"][0])]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], doc_hash: str) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata(doc_hash) for c in chunks],
            embeddings=embeddings,
        )
        self.version += 1

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        self.version += 1


def _to_chunks(ids, documents, metadatas) -> list[Chunk]:
    chunks = []
    for chunk_id, text, meta in zip(ids, documents or [], metadatas or []):
        if not text:
            continue
        meta = meta or {}
        chunks.append(
            Chunk(
                id=chunk_id,
                text=text,
                source=meta.get("source", "Unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk", 0)),
            )
        )
    return chunks
