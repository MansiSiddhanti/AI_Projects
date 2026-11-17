from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .chunking import chunk_text
from .config import RetrievalMode, Settings, get_settings
from .embeddings import Embedder, Reranker
from .intent import is_summary_request, match_source
from .llm import ChatModel, Message
from .loaders import Page, load_pdf, load_pdf_bytes, sha256_bytes
from .models import Answer, Chunk, DocumentInfo, IngestResult, ScoredChunk
from .retrieval import Retriever
from .store import VectorStore

ProgressCallback = Callable[[int, int, str], None]

_CITATION = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers found in an answer."""
    cited = sorted({int(n) for n in _CITATION.findall(text)})
    return [n for n in cited if 1 <= n <= n_sources], [n for n in cited if not 1 <= n <= n_sources]


@dataclass
class RetrievalOptions:
    top_k: int
    mode: RetrievalMode
    rerank: bool
    sources: list[str] | None = None


@dataclass
class StreamingAnswer:
    question: str
    standalone_question: str
    sources: list[ScoredChunk]
    tokens: Iterator[str]
    timings: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        llm: ChatModel,
        reranker_factory: Callable[[], Reranker] | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.retriever = Retriever(
            store,
            embedder,
            reranker_factory=reranker_factory,
            candidate_k=settings.candidate_k,
            rrf_k=settings.rrf_k,
        )
        self._summary_cache_path = Path(settings.database_dir) / "summaries.json"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RAGPipeline:
        from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
        from .llm import OllamaChat

        settings = settings or get_settings()
        return cls(
            settings=settings,
            embedder=SentenceTransformerEmbedder(
                settings.embedding_model, settings.embedding_query_prefix
            ),
            store=VectorStore(
                settings.collection_name,
                settings.embedding_model,
                path=settings.database_dir,
            ),
            llm=OllamaChat(
                settings.llm_model,
                host=settings.ollama_host,
                temperature=settings.temperature,
                context_window=settings.llm_context_window,
            ),
            reranker_factory=lambda: CrossEncoderReranker(settings.reranker_model),
        )

    def chunk_pages(self, source: str, pages: list[Page]) -> list[Chunk]:
        chunks = []
        for page in pages:
            pieces = chunk_text(
                page.text,
                max_tokens=self.settings.chunk_max_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
                count_tokens=self.embedder.count_tokens,
            )
            for i, text in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=f"{source}::p{page.number}::c{i}",
                        text=text,
                        source=source,
                        page=page.number,
                        chunk_index=i,
                    )
                )
        return chunks

    def ingest_pages(
        self, source: str, pages: list[Page], doc_hash: str, force: bool = False
    ) -> IngestResult:
        if not force and self.store.source_hash(source) == doc_hash:
            return IngestResult(source, "unchanged")

        chunks = self.chunk_pages(source, pages)
        # old version may have had more chunks, clear it out first
        self.store.delete_source(source)
        if not chunks:
            return IngestResult(source, "empty", pages=len(pages))

        self.store.add(chunks, self.embedder.embed_documents([c.text for c in chunks]), doc_hash)
        return IngestResult(source, "indexed", chunks=len(chunks), pages=len(pages))

    def ingest_bytes(self, source: str, data: bytes, force: bool = False) -> IngestResult:
        return self.ingest_pages(source, load_pdf_bytes(data), sha256_bytes(data), force)

    def ingest_file(self, path: str | Path, force: bool = False) -> IngestResult:
        pages, doc_hash = load_pdf(path)
        return self.ingest_pages(Path(path).name, pages, doc_hash, force)

    def ingest_directory(
        self, directory: str | Path | None = None, force: bool = False
    ) -> list[IngestResult]:
        directory = Path(directory or self.settings.documents_dir)
        return [self.ingest_file(p, force) for p in sorted(directory.glob("*.pdf"))]

    def documents(self) -> list[DocumentInfo]:
        return self.store.documents()

    def delete_document(self, source: str) -> None:
        self.store.delete_source(source)

    def default_options(self, **overrides) -> RetrievalOptions:
        opts = RetrievalOptions(
            top_k=self.settings.top_k,
            mode=self.settings.retrieval_mode,
            rerank=self.settings.use_reranker,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(opts, key, value)
        return opts

    def retrieve(self, query: str, options: RetrievalOptions | None = None) -> list[ScoredChunk]:
        o = options or self.default_options()
        return self.retriever.retrieve(
            query, top_k=o.top_k, mode=o.mode, rerank=o.rerank, sources=o.sources
        )

    def _recent_history(self, history: list[Message] | None) -> list[Message]:
        turns = self.settings.history_turns
        return [m for m in (history or []) if m["role"] in ("user", "assistant")][-2 * turns :]

    def condense_question(self, question: str, history: list[Message] | None) -> str:
        # "what about its limitations?" is useless as a search query on its own,
        # so ask the LLM to rewrite it using the chat history
        recent = self._recent_history(history)
        if not recent or not self.settings.condense_followups:
            return question
        prompt = prompts.CONDENSE_PROMPT.format(
            history=prompts.format_history(recent), question=question
        )
        rewritten = self.llm.chat([{"role": "user", "content": prompt}]).strip().strip('"')
        return rewritten.splitlines()[0] if rewritten else question

    def prepare(
        self,
        question: str,
        history: list[Message] | None = None,
        options: RetrievalOptions | None = None,
    ) -> tuple[str, list[ScoredChunk], dict[str, float]]:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        standalone = self.condense_question(question, history)
        timings["condense"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        passages = self.retrieve(standalone, options)
        timings["retrieve"] = time.perf_counter() - t0
        return standalone, passages, timings

    def stream_answer(
        self,
        question: str,
        history: list[Message] | None = None,
        options: RetrievalOptions | None = None,
    ) -> StreamingAnswer:
        standalone, passages, timings = self.prepare(question, history, options)
        if not passages:
            tokens = iter(
                [
                    "I couldn't find any indexed passages to answer from. "
                    "Upload and index some documents first."
                ]
            )
        else:
            messages = prompts.build_qa_messages(
                standalone, passages, self._recent_history(history)
            )
            tokens = self.llm.stream(messages)
        return StreamingAnswer(question, standalone, passages, tokens, timings)

    def answer(
        self,
        question: str,
        history: list[Message] | None = None,
        options: RetrievalOptions | None = None,
    ) -> Answer:
        streamed = self.stream_answer(question, history, options)
        t0 = time.perf_counter()
        text = "".join(streamed.tokens).strip()
        streamed.timings["generate"] = time.perf_counter() - t0
        valid, invalid = parse_citations(text, len(streamed.sources))
        return Answer(
            question=question,
            standalone_question=streamed.standalone_question,
            text=text,
            sources=streamed.sources,
            cited=valid,
            invalid_citations=invalid,
            timings=streamed.timings,
        )

    def route(self, question: str, selected_source: str | None = None) -> str | None:
        if not is_summary_request(question):
            return None
        names = [d.source for d in self.documents()]
        return (
            match_source(question, names)
            or selected_source
            or (names[0] if len(names) == 1 else None)
        )

    def _load_summary_cache(self) -> dict:
        try:
            return json.loads(self._summary_cache_path.read_text())
        except (OSError, ValueError):
            return {}

    def _summary_key(self, source: str) -> str:
        return f"{source}|{self.store.source_hash(source)}|{self.llm.name}"

    def cached_summary(self, source: str) -> str | None:
        return self._load_summary_cache().get(self._summary_key(source))

    def summarize(
        self, source: str, progress: ProgressCallback | None = None, use_cache: bool = True
    ) -> str:
        """Hierarchical map-reduce summary of an entire document.

        Chunks are packed into token-budgeted batches (map), then partial
        summaries are merged in groups until one remains (reduce). Results are
        cached per document hash + model, since this is slow on CPU.
        """
        key = self._summary_key(source)
        if use_cache and (cached := self.cached_summary(source)):
            return cached

        chunks = self.store.source_chunks(source)
        if not chunks:
            raise ValueError(f"No indexed text found for '{source}'.")

        batches: list[list[str]] = [[]]
        budget = 0
        for chunk in chunks:
            n = self.embedder.count_tokens(chunk.text)
            if batches[-1] and budget + n > self.settings.summary_map_tokens:
                batches.append([])
                budget = 0
            batches[-1].append(chunk.text)
            budget += n

        group = max(2, self.settings.summary_reduce_group)
        total_steps = len(batches) + _reduce_steps(len(batches), group)
        step = 0

        def report(message: str) -> None:
            nonlocal step
            step += 1
            if progress:
                progress(step, total_steps, message)

        summaries = []
        for i, batch in enumerate(batches, start=1):
            prompt = prompts.SUMMARY_MAP_PROMPT.format(
                source=source, part=i, total=len(batches), text="\n\n".join(batch)
            )
            summaries.append(self.llm.chat([{"role": "user", "content": prompt}]))
            report(f"Summarized part {i}/{len(batches)}")

        # always do at least one reduce so we get the sectioned format
        while True:
            merged = []
            for start in range(0, len(summaries), group):
                parts = summaries[start : start + group]
                text = "\n\n".join(f"Part {j}:\n{s}" for j, s in enumerate(parts, start=1))
                prompt = prompts.SUMMARY_REDUCE_PROMPT.format(source=source, text=text)
                merged.append(self.llm.chat([{"role": "user", "content": prompt}]))
                report("Merging partial summaries")
            summaries = merged
            if len(summaries) == 1:
                break

        result = summaries[0]
        cache = self._load_summary_cache()
        cache[key] = result
        self._summary_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_cache_path.write_text(json.dumps(cache, indent=2))
        return result


def _reduce_steps(n: int, group: int) -> int:
    steps = 0
    while True:
        n = -(-n // group)
        steps += n
        if n == 1:
            return steps
