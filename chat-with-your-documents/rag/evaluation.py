# Retrieval eval. The LLM writes a question for a random chunk, and that
# chunk's (source, page) is the expected hit. Page level since chunks overlap.
import json
import random
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import prompts
from .pipeline import RAGPipeline, RetrievalOptions

# skip reference lists / author bios, questions from those are useless
_REFERENCE_NOISE = re.compile(
    r"\bet al\.|\bdoi\b|\bvol\.|\bpp\.|\bIEEE Access\b|\breceived the\b|\bhis research\b"
    r"|\bher research\b|\bdigital object identifier\b",
    re.IGNORECASE,
)

CONFIGS: dict[str, dict] = {
    "dense": {"mode": "dense", "rerank": False},
    "bm25": {"mode": "bm25", "rerank": False},
    "hybrid (RRF)": {"mode": "hybrid", "rerank": False},
    "hybrid + rerank": {"mode": "hybrid", "rerank": True},
}


@dataclass
class EvalItem:
    question: str
    source: str
    page: int
    chunk_id: str


def _usable(text: str, n_tokens: int) -> bool:
    return n_tokens >= 120 and len(_REFERENCE_NOISE.findall(text)) <= 1


def generate_dataset(
    pipeline: RAGPipeline,
    n: int,
    out_path: str | Path,
    seed: int = 7,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[EvalItem]:
    rng = random.Random(seed)
    chunks = [
        c
        for c in pipeline.store.all_chunks()
        if _usable(c.text, pipeline.embedder.count_tokens(c.text))
    ]
    # round robin over papers, otherwise Paper_02 (44 pages) dominates
    by_source: dict[str, list] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    for group in by_source.values():
        rng.shuffle(group)
    sample = []
    while len(sample) < n and any(by_source.values()):
        for group in by_source.values():
            if group and len(sample) < n:
                sample.append(group.pop())

    items = []
    for i, chunk in enumerate(sample, start=1):
        prompt = prompts.QUESTION_GEN_PROMPT.format(text=chunk.text)
        question = pipeline.llm.chat([{"role": "user", "content": prompt}])
        question = question.strip().strip('"').splitlines()[0].strip()
        if question.endswith("?"):
            items.append(EvalItem(question, chunk.source, chunk.page, chunk.id))
        if progress:
            progress(i, len(sample), question)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(asdict(item)) + "\n" for item in items))
    return items


def load_dataset(path: str | Path) -> list[EvalItem]:
    lines = Path(path).read_text().splitlines()
    return [EvalItem(**json.loads(line)) for line in lines if line.strip()]


def evaluate(
    pipeline: RAGPipeline,
    dataset: list[EvalItem],
    k: int = 5,
    configs: dict[str, dict] | None = None,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, cfg in (configs or CONFIGS).items():
        hits1 = hitsk = rr = 0.0
        latencies = []
        for item in dataset:
            t0 = time.perf_counter()
            retrieved = pipeline.retrieve(
                item.question, RetrievalOptions(top_k=k, mode=cfg["mode"], rerank=cfg["rerank"])
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            rank = next(
                (
                    i
                    for i, r in enumerate(retrieved, start=1)
                    if r.chunk.source == item.source and r.chunk.page == item.page
                ),
                None,
            )
            if rank:
                hits1 += rank == 1
                hitsk += 1
                rr += 1 / rank
        n = len(dataset) or 1
        results[name] = {
            "hit@1": hits1 / n,
            f"hit@{k}": hitsk / n,
            f"mrr@{k}": rr / n,
            "p50_ms": statistics.median(latencies) if latencies else 0.0,
        }
    return results


def to_markdown(results: dict[str, dict[str, float]], n_questions: int) -> str:
    metrics = list(next(iter(results.values())).keys())
    header = "| Retriever | " + " | ".join(m.replace("_", " ") for m in metrics) + " |"
    rule = "|---|" + "---:|" * len(metrics)
    rows = []
    for name, values in results.items():
        cells = [f"{v:.0f}" if m.endswith("_ms") else f"{v:.3f}" for m, v in values.items()]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows, "", f"_{n_questions} questions, page-level match._"])
