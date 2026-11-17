import argparse
import json
import sys
from pathlib import Path

from .config import get_settings


def _pipeline():
    from .pipeline import RAGPipeline

    return RAGPipeline.from_settings()


def cmd_ingest(args) -> int:
    pipeline = _pipeline()
    target = Path(args.path or pipeline.settings.documents_dir)
    files = [target] if target.is_file() else sorted(target.glob("*.pdf"))
    if not files:
        print(f"No PDFs found in {target}")
        return 1
    for path in files:
        result = pipeline.ingest_file(path, force=args.force)
        detail = f"{result.chunks} chunks from {result.pages} pages" if result.chunks else ""
        print(f"{result.status:>9}  {result.source}  {detail}")
    print(f"\nTotal chunks in index: {pipeline.store.count()}")
    return 0


def _print_answer(pipeline, question, history):
    streamed = pipeline.stream_answer(question, history)
    if streamed.standalone_question != question:
        print(f"(searching for: {streamed.standalone_question})")
    print()
    parts = []
    for token in streamed.tokens:
        parts.append(token)
        print(token, end="", flush=True)
    print("\n\nSources:")
    for i, s in enumerate(streamed.sources, start=1):
        print(f"  [{i}] {s.chunk.source}, p.{s.chunk.page}  (score {s.score:.3f})")
    return "".join(parts)


def cmd_ask(args) -> int:
    pipeline = _pipeline()
    question = " ".join(args.question)
    if source := pipeline.route(question):
        print(pipeline.summarize(source, progress=lambda i, n, m: print(f"  [{i}/{n}] {m}")))
    else:
        _print_answer(pipeline, question, [])
    return 0


def cmd_chat(args) -> int:
    pipeline = _pipeline()
    ok, message = pipeline.llm.health()
    print(message)
    if not ok:
        return 1
    print(f"{pipeline.store.count()} chunks indexed. Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        if source := pipeline.route(question):
            answer = pipeline.summarize(source, progress=lambda i, n, m: print(f"  [{i}/{n}] {m}"))
            print(answer)
        else:
            answer = _print_answer(pipeline, question, history)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        print()


def cmd_eval(args) -> int:
    from . import evaluation

    pipeline = _pipeline()
    dataset_path = Path(args.dataset)
    if args.generate or not dataset_path.exists():
        print(f"Generating {args.n} synthetic questions with {pipeline.llm.name}...")
        evaluation.generate_dataset(
            pipeline,
            args.n,
            dataset_path,
            progress=lambda i, n, q: print(f"  [{i}/{n}] {q}"),
        )
    dataset = evaluation.load_dataset(dataset_path)
    print(f"\nEvaluating {len(dataset)} questions...")
    results = evaluation.evaluate(pipeline, dataset, k=args.k)
    table = evaluation.to_markdown(results, len(dataset))
    print("\n" + table)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("rag.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Chat with your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Index PDFs (skips unchanged files).")
    p.add_argument("path", nargs="?", help="PDF file or folder (default: documents/)")
    p.add_argument("--force", action="store_true", help="Re-index even if unchanged.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask a single question.")
    p.add_argument("question", nargs="+")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Interactive multi-turn chat in the terminal.")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="Benchmark retrieval strategies.")
    p.add_argument("--dataset", default="eval/questions.jsonl")
    p.add_argument("--generate", action="store_true", help="(Re)generate the dataset.")
    p.add_argument("-n", type=int, default=40, help="Questions to generate.")
    p.add_argument("-k", type=int, default=get_settings().top_k)
    p.add_argument("--output", default="eval/results.json")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="Run the REST API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
