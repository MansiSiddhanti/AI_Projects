# fakes so tests don't need ollama or model downloads
import hashlib
import math
import uuid
from collections.abc import Iterator

import chromadb
import pytest

from rag.bm25 import tokenize
from rag.config import Settings
from rag.pipeline import RAGPipeline
from rag.store import VectorStore


class FakeEmbedder:
    # hashed bag of words, good enough for similar text -> similar vector

    name = "fake-embedder"
    dim = 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    name = "fake-llm"

    def __init__(self, reply: str = "Answer citing [1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages) -> Iterator[str]:
        self.calls.append(messages)
        yield from self.reply.split(" ")[:1]
        for word in self.reply.split(" ")[1:]:
            yield " " + word

    def health(self):
        return True, "fake ready"


class FakeReranker:
    def score(self, query, passages):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) for p in passages]


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(
        collection_name=f"test-{uuid.uuid4().hex[:8]}",
        embedding_model=FakeEmbedder.name,
        client=chromadb.EphemeralClient(),
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_dir=tmp_path,
        chunk_max_tokens=40,
        chunk_overlap_tokens=8,
        candidate_k=10,
        top_k=3,
        _env_file=None,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(settings, store, llm) -> RAGPipeline:
    return RAGPipeline(
        settings=settings,
        embedder=FakeEmbedder(),
        store=store,
        llm=llm,
        reranker_factory=FakeReranker,
    )
