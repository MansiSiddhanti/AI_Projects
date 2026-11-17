# run with: python -m rag serve  (swagger docs at /docs)
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = []
    sources: list[str] | None = Field(default=None, description="Restrict search to these PDFs")
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    rerank: bool | None = None
    stream: bool = False


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline.from_settings()
        yield

    app = FastAPI(
        title="Chat with Your Documents",
        description="Local, private RAG over your PDFs (hybrid search + reranking + Ollama).",
        version="2.0.0",
        lifespan=lifespan,
    )

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health(p: RAGPipeline = Depends(get_pipeline)):
        llm_ok, message = p.llm.health() if hasattr(p.llm, "health") else (True, "ok")
        return {
            "status": "ok" if llm_ok else "degraded",
            "llm": message,
            "chunks_indexed": p.store.count(),
        }

    @app.get("/documents")
    def list_documents(p: RAGPipeline = Depends(get_pipeline)):
        return [d.__dict__ for d in p.documents()]

    @app.post("/documents", status_code=201)
    async def upload_documents(
        files: list[UploadFile] = File(...), p: RAGPipeline = Depends(get_pipeline)
    ):
        results = []
        for f in files:
            name = Path(f.filename or "").name
            if not name.lower().endswith(".pdf"):
                raise HTTPException(415, f"'{name}' is not a PDF")
            results.append(p.ingest_bytes(name, await f.read()).__dict__)
        return results

    @app.delete("/documents/{source}", status_code=204)
    def delete_document(source: str, p: RAGPipeline = Depends(get_pipeline)):
        if source not in {d.source for d in p.documents()}:
            raise HTTPException(404, f"'{source}' is not indexed")
        p.delete_document(source)

    @app.post("/documents/{source}/summary")
    def summarize(source: str, p: RAGPipeline = Depends(get_pipeline)):
        if source not in {d.source for d in p.documents()}:
            raise HTTPException(404, f"'{source}' is not indexed")
        return {"source": source, "summary": p.summarize(source)}

    @app.post("/ask")
    def ask(req: AskRequest, p: RAGPipeline = Depends(get_pipeline)):
        history = [m.model_dump() for m in req.history]
        options = p.default_options(
            top_k=req.top_k, mode=req.mode, rerank=req.rerank, sources=req.sources
        )
        if not req.stream:
            return p.answer(req.question, history, options).to_dict()

        streamed = p.stream_answer(req.question, history, options)

        def events() -> Iterator[str]:
            # ndjson: sources first, then tokens, then done
            yield (
                json.dumps(
                    {
                        "type": "sources",
                        "standalone_question": streamed.standalone_question,
                        "sources": [s.to_dict() for s in streamed.sources],
                    }
                )
                + "\n"
            )
            for token in streamed.tokens:
                yield json.dumps({"type": "token", "content": token}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


def __getattr__(name: str):
    # so `uvicorn rag.api:app` works without loading models on import (tests)
    if name == "app":
        return create_app()
    raise AttributeError(name)
