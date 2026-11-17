from rag.chunking import chunk_text, split_sentences, whitespace_tokens
from rag.loaders import clean_text

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(60))


def test_chunks_respect_token_budget():
    chunks = chunk_text(TEXT, max_tokens=30, overlap_tokens=6)
    assert len(chunks) > 1
    assert all(whitespace_tokens(c) <= 30 for c in chunks)


def test_chunks_cover_all_sentences_in_order():
    chunks = chunk_text(TEXT, max_tokens=30, overlap_tokens=0)
    assert " ".join(chunks) == TEXT


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, max_tokens=30, overlap_tokens=12)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev)[-1]
        assert nxt.startswith(last_sentence)


def test_oversized_sentence_is_split_on_words():
    long_sentence = " ".join(["word"] * 95)
    chunks = chunk_text(long_sentence, max_tokens=20)
    assert [whitespace_tokens(c) for c in chunks] == [20, 20, 20, 20, 15]


def test_short_and_empty_text():
    assert chunk_text("Just one sentence.", max_tokens=50) == ["Just one sentence."]
    assert chunk_text("   ", max_tokens=50) == []


def test_clean_text_dehyphenates_and_collapses_whitespace():
    assert clean_text("transfor-\nmer   models\n\nare  COVID-19 ready") == (
        "transformer models are COVID-19 ready"
    )
