# Chat with Your Documents

A local RAG app for asking questions about PDFs. Everything runs on your own
machine: embeddings with sentence-transformers, vector search with ChromaDB, and
answers from Llama 3.2 through Ollama. I built it to work with a set of IEEE
survey papers, but it works with any text-based PDF.

There is a Streamlit chat UI, a FastAPI backend, and a small CLI, all sharing
the same pipeline in `rag/`.

## What it does

- Answers questions with citations back to the PDF and page they came from
- Handles follow-up questions by rewriting them into standalone queries using
  the chat history
- Summarizes a whole paper on request (map-reduce over the chunks; results are
  cached because this is slow on CPU)
- Re-indexes only PDFs that changed (files are tracked by SHA-256 hash)
- Lets you restrict search to specific documents

## How retrieval works

1. PDFs are split per page, then into chunks of at most 220 tokens on sentence
   boundaries with a 40-token overlap.
2. A query is run against both the vector index (cosine similarity) and a BM25
   keyword index, taking the top 20 from each.
3. The two lists are merged with Reciprocal Rank Fusion.
4. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the merged candidates and
   the top 5 go to the LLM.

A few notes on why it's set up this way:

- **Chunk size.** My first version used 2,000-character chunks. When I checked,
  those were around 400 tokens, but `all-MiniLM-L6-v2` only reads the first 256,
  so a large part of every chunk was never embedded. Chunks are now measured with
  the embedding model's own tokenizer.
- **BM25 alongside vectors.** Embeddings are good with paraphrases but weak on
  exact terms like acronyms or method names (SMOTE, LoRA, etc.). BM25 covers that.
  RRF merges by rank, so I don't have to normalize two different score scales.
- **Reranking.** The cross-encoder is much more accurate than comparing
  embeddings, but too slow to run over the whole index, so it only sees the
  merged top 20.

## Evaluation

To compare retrieval setups I generated 50 test questions with the LLM: pick a
random chunk, ask Llama 3.2 to write a question that chunk answers, and record
which paper and page it came from. A retriever gets a hit if that page shows up
in its top 5. (`python -m rag eval`, code in `rag/evaluation.py`.)

| Setup | hit@1 | hit@5 | MRR@5 | median latency |
|---|---:|---:|---:|---:|
| Original version (dense, 2000-char chunks) | 0.38 | 0.80 | 0.53 | 25 ms |
| Dense only | 0.56 | 0.74 | 0.63 | 17 ms |
| BM25 only | 0.84 | 1.00 | 0.92 | 3 ms |
| Hybrid (RRF) | 0.74 | 0.96 | 0.83 | 21 ms |
| Hybrid + reranker (default) | **0.92** | **1.00** | **0.95** | 976 ms |

Measured on CPU (no GPU) over the 5 papers in this repo (110 pages, 777 chunks).

Some caveats. The questions are generated from the text itself, so they
tend to reuse the paper's exact wording, which flatters BM25. Real questions
are usually phrased differently, which is why I kept the dense retriever in
the mix instead of switching to BM25 alone. 50 questions is also a small set.
The reranker adds about a second per query on CPU, which is still small next to
the LLM's generation time.

## Setup

You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull llama3.2

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Put your PDFs in `documents/` and index them:

```bash
python -m rag ingest
```

Then run one of:

```bash
streamlit run app.py     # web UI on :8501
python -m rag serve      # REST API on :8000, docs at /docs
python -m rag chat       # terminal chat
```

### Docker

```bash
docker compose up -d ollama
docker compose run --rm ollama-pull
docker compose up ui api
```

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | Ollama status and number of indexed chunks |
| GET | `/documents` | List indexed documents |
| POST | `/documents` | Upload PDFs (multipart) |
| DELETE | `/documents/{name}` | Remove a document |
| POST | `/documents/{name}/summary` | Summarize a document |
| POST | `/ask` | Ask a question (`"stream": true` for NDJSON streaming) |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which techniques address class imbalance?", "sources": ["Paper_04.pdf"]}'
```

## Configuration

Settings are in `rag/config.py`. Any of them can be overridden with an
environment variable prefixed with `RAG_` or in a `.env` file. See `.env.example`.

| Variable | Default |
|---|---|
| `RAG_LLM_MODEL` | `llama3.2` |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `RAG_RETRIEVAL_MODE` | `hybrid` (`dense`, `bm25`, `hybrid`) |
| `RAG_USE_RERANKER` | `true` |
| `RAG_TOP_K` | `5` |
| `RAG_CHUNK_MAX_TOKENS` | `220` |

If you change the embedding model, also change `RAG_COLLECTION_NAME` (or delete
`chroma_db/`). The app refuses to mix embeddings from different models in one
collection.

## Project layout

```
rag/
  config.py       settings
  loaders.py      PDF text extraction and cleanup
  chunking.py     chunker
  embeddings.py   embedding model and cross-encoder
  store.py        ChromaDB wrapper
  bm25.py         BM25 keyword index
  retrieval.py    dense / BM25 / hybrid retrieval and reranking
  prompts.py      prompt templates
  pipeline.py     ingestion, Q&A, summaries
  intent.py       detects summary requests
  evaluation.py   retrieval benchmark
  api.py          FastAPI app
  cli.py          command line
app.py            Streamlit UI
tests/
```

## Tests

```bash
pytest
ruff check .
```

The tests use small fake embedding and LLM classes, so they run in a couple of
seconds without downloading models or running Ollama. CI runs the same on every
push.

## Things I'd like to add

- Scoring answer quality, not just retrieval (faithfulness, relevance)
- OCR for scanned PDFs
- Better handling of tables
- Showing the cited passage highlighted in the PDF
