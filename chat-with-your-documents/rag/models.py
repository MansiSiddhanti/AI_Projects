from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int

    def metadata(self, doc_hash: str = "") -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "chunk": self.chunk_index,
            "doc_hash": doc_hash,
        }


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Per-stage scores, e.g. {"dense": 0.71, "bm25": 9.2, "rerank": 4.1}
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


@dataclass
class DocumentInfo:
    source: str
    pages: int
    chunks: int
    doc_hash: str


@dataclass
class IngestResult:
    source: str
    status: str  # "indexed" | "unchanged" | "empty"
    chunks: int = 0
    pages: int = 0


@dataclass
class Answer:
    question: str
    standalone_question: str
    text: str
    sources: list[ScoredChunk]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "answer": self.text,
            "sources": [s.to_dict() for s in self.sources],
            "cited": self.cited,
            "invalid_citations": self.invalid_citations,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }
