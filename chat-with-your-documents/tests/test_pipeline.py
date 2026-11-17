import pytest

from rag.intent import is_summary_request, match_source
from rag.loaders import Page
from rag.pipeline import parse_citations

PAGES = [
    Page(1, "Agentic AI systems plan and act autonomously. They use tools and memory."),
    Page(2, "Risks of agentic AI include misalignment and misuse. Governance is needed."),
]


def test_ingest_is_idempotent_and_detects_changes(pipeline):
    first = pipeline.ingest_pages("agents.pdf", PAGES, doc_hash="v1")
    assert first.status == "indexed" and first.chunks >= 2

    assert pipeline.ingest_pages("agents.pdf", PAGES, doc_hash="v1").status == "unchanged"

    changed = pipeline.ingest_pages("agents.pdf", PAGES[:1], doc_hash="v2")
    assert changed.status == "indexed"
    # The page-2 chunks from v1 must be gone, not left behind as stale data.
    assert {c.page for c in pipeline.store.source_chunks("agents.pdf")} == {1}


def test_documents_and_delete(pipeline):
    pipeline.ingest_pages("a.pdf", PAGES, "h1")
    pipeline.ingest_pages("b.pdf", PAGES[:1], "h2")
    docs = {d.source: d for d in pipeline.documents()}
    assert docs["a.pdf"].pages == 2 and docs["b.pdf"].pages == 1
    pipeline.delete_document("a.pdf")
    assert [d.source for d in pipeline.documents()] == ["b.pdf"]


def test_answer_returns_sources_and_validated_citations(pipeline, llm):
    pipeline.ingest_pages("agents.pdf", PAGES, "h")
    llm.reply = "Agents use tools [1] and face risks [2][9]."
    answer = pipeline.answer("What risks do agentic AI systems have?")
    assert answer.sources
    assert answer.cited == [1, 2]
    assert answer.invalid_citations == [9]
    # The prompt sent to the LLM contains the numbered sources.
    assert "[1] (from agents.pdf" in llm.calls[-1][-1]["content"]


def test_followup_is_condensed_using_history(pipeline, llm):
    pipeline.ingest_pages("agents.pdf", PAGES, "h")
    llm.reply = "What are the risks of agentic AI?"
    history = [
        {"role": "user", "content": "Tell me about agentic AI"},
        {"role": "assistant", "content": "It plans and acts."},
    ]
    answer = pipeline.answer("What are its risks?", history)
    assert answer.standalone_question == "What are the risks of agentic AI?"


def test_no_documents_gives_helpful_message(pipeline):
    assert "Upload and index" in pipeline.answer("anything?").text


def test_summarize_map_reduce_and_cache(pipeline, llm):
    pipeline.settings.summary_map_tokens = 15  # force several map batches
    pipeline.ingest_pages("agents.pdf", PAGES, "h")
    llm.reply = "A summary."
    steps = []
    assert pipeline.summarize("agents.pdf", progress=lambda i, n, m: steps.append((i, n))) == (
        "A summary."
    )
    assert steps[-1][0] == steps[-1][1]  # progress reaches 100%
    calls = len(llm.calls)
    pipeline.summarize("agents.pdf")
    assert len(llm.calls) == calls  # served from cache


def test_summarize_unknown_document(pipeline):
    with pytest.raises(ValueError):
        pipeline.summarize("missing.pdf")


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the paper",  # regression: the old regex required a leading space
        "Can you summarize Paper_02?",
        "What are the key findings?",
        "tl;dr please",
        "What is this paper about?",
    ],
)
def test_summary_intent_positive(question):
    assert is_summary_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is SMOTE?",
        "How does federated learning protect privacy?",
    ],
)
def test_summary_intent_negative(question):
    assert not is_summary_request(question)


def test_match_source_variants():
    sources = ["Paper_01.pdf", "Paper_03.pdf"]
    assert match_source("summarize Paper_03.pdf", sources) == "Paper_03.pdf"
    assert match_source("summary of paper 3 please", sources) == "Paper_03.pdf"
    assert match_source("summary of paper 13", sources) is None


def test_parse_citations():
    assert parse_citations("x [1] y [3][2] z [7]", 3) == ([1, 2, 3], [7])
