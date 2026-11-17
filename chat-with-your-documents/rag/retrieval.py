from collections.abc import Callable

from .bm25 import BM25
from .config import RetrievalMode
from .embeddings import Embedder, Reranker
from .models import Chunk, ScoredChunk
from .store import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    # RRF: score(d) = sum over rankings of 1 / (k + rank)
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker_factory: Callable[[], Reranker] | None = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self._reranker_factory = reranker_factory
        self._reranker: Reranker | None = None
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_version: tuple[int, int] | None = None

    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self._reranker_factory is not None:
            self._reranker = self._reranker_factory()
        return self._reranker

    def _keyword_index(self) -> tuple[BM25, list[Chunk]]:
        # check count too, the API and UI may both write to the same db
        key = (self.store.version, self.store.count())
        if self._bm25 is None or self._bm25_version != key:
            self._bm25_chunks = self.store.all_chunks()
            self._bm25 = BM25([c.text for c in self._bm25_chunks])
            self._bm25_version = key
        return self._bm25, self._bm25_chunks

    def _dense(self, query: str, n: int, sources: list[str] | None) -> list[ScoredChunk]:
        hits = self.store.query(self.embedder.embed_query(query), n, sources)
        return [ScoredChunk(c, s, {"dense": s}) for c, s in hits]

    def _keyword(self, query: str, n: int, sources: list[str] | None) -> list[ScoredChunk]:
        index, chunks = self._keyword_index()
        wanted = set(sources or [])
        allowed = {i for i, c in enumerate(chunks) if c.source in wanted} if sources else None
        return [ScoredChunk(chunks[i], s, {"bm25": s}) for i, s in index.top_n(query, n, allowed)]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        sources: list[str] | None = None,
    ) -> list[ScoredChunk]:
        query = query.strip()
        if not query or self.store.count() == 0:
            return []

        n = max(self.candidate_k, top_k)

        if mode == "dense":
            candidates = self._dense(query, n, sources)
        elif mode == "bm25":
            candidates = self._keyword(query, n, sources)
        elif mode == "hybrid":
            dense = self._dense(query, n, sources)
            keyword = self._keyword(query, n, sources)
            fused = reciprocal_rank_fusion(
                [[h.chunk.id for h in dense], [h.chunk.id for h in keyword]], k=self.rrf_k
            )
            merged: dict[str, ScoredChunk] = {}
            for hit in dense + keyword:
                entry = merged.setdefault(hit.chunk.id, ScoredChunk(hit.chunk, 0.0, {}))
                entry.scores.update(hit.scores)
            for chunk_id, entry in merged.items():
                entry.score = entry.scores["rrf"] = fused[chunk_id]
            candidates = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:n]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        if rerank and candidates and self.reranker is not None:
            rerank_scores = self.reranker.score(query, [h.chunk.text for h in candidates])
            for hit, score in zip(candidates, rerank_scores):
                hit.score = hit.scores["rerank"] = score
            candidates.sort(key=lambda h: h.score, reverse=True)

        return candidates[:top_k]
