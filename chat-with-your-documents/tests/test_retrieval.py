import pytest

from rag.bm25 import BM25
from rag.models import Chunk
from rag.retrieval import Retriever, reciprocal_rank_fusion

from .conftest import FakeEmbedder, FakeReranker

DOCS = {
    "ml.pdf": [
        "SMOTE oversampling creates synthetic minority class examples for imbalanced data.",
        "Undersampling removes majority class examples to rebalance the dataset.",
    ],
    "iot.pdf": [
        "Edge computing moves inference closer to IoT sensors to reduce latency.",
        "Federated learning trains models across IoT devices without sharing raw data.",
    ],
}


@pytest.fixture
def populated(store):
    emb = FakeEmbedder()
    for source, texts in DOCS.items():
        chunks = [Chunk(f"{source}::{i}", t, source, i + 1, 0) for i, t in enumerate(texts)]
        store.add(chunks, emb.embed_documents(texts), doc_hash="h")
    return store


def test_bm25_ranks_keyword_match_first():
    bm25 = BM25(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    top = bm25.top_n("cat", 3)
    assert {i for i, _ in top} == {0, 2}
    assert bm25.top_n("zebra", 3) == []


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == fused["b"] > fused["c"] == fused["d"]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_each_mode_finds_relevant_chunk(populated, mode):
    retriever = Retriever(populated, FakeEmbedder())
    hits = retriever.retrieve("SMOTE oversampling minority", top_k=2, mode=mode, rerank=False)
    assert hits[0].chunk.source == "ml.pdf"
    assert hits[0].chunk.page == 1


def test_hybrid_records_component_scores_and_reranks(populated):
    retriever = Retriever(populated, FakeEmbedder(), reranker_factory=FakeReranker)
    hits = retriever.retrieve("federated learning IoT devices", top_k=2, mode="hybrid")
    assert hits[0].chunk.id == "iot.pdf::1"
    assert {"dense", "rrf", "rerank"} <= hits[0].scores.keys()


def test_source_filter_restricts_results(populated):
    retriever = Retriever(populated, FakeEmbedder())
    hits = retriever.retrieve(
        "examples data", top_k=4, mode="hybrid", rerank=False, sources=["iot.pdf"]
    )
    assert hits and {h.chunk.source for h in hits} == {"iot.pdf"}


def test_bm25_index_refreshes_after_writes(populated):
    retriever = Retriever(populated, FakeEmbedder())
    assert retriever.retrieve("quantum", mode="bm25", rerank=False) == []
    populated.add(
        [Chunk("q::0", "quantum computing basics", "q.pdf", 1, 0)],
        FakeEmbedder().embed_documents(["quantum computing basics"]),
        "h",
    )
    assert retriever.retrieve("quantum", mode="bm25", rerank=False)[0].chunk.source == "q.pdf"


def test_empty_query_or_index(store):
    retriever = Retriever(store, FakeEmbedder())
    assert retriever.retrieve("anything") == []
    assert retriever.retrieve("   ") == []
