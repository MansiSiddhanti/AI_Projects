import json

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.loaders import Page


@pytest.fixture
def client(pipeline):
    pipeline.ingest_pages("agents.pdf", [Page(1, "Agentic AI systems use tools.")], "h")
    with TestClient(create_app(pipeline)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 1


def test_list_and_delete_documents(client):
    assert [d["source"] for d in client.get("/documents").json()] == ["agents.pdf"]
    assert client.delete("/documents/agents.pdf").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete("/documents/agents.pdf").status_code == 404


def test_upload_rejects_non_pdf(client):
    r = client.post("/documents", files={"files": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_ask_json(client):
    body = client.post("/ask", json={"question": "What do agents use?"}).json()
    assert body["answer"] == "Answer citing [1]."
    assert body["sources"][0]["source"] == "agents.pdf"
    assert body["cited"] == [1]


def test_ask_streaming_ndjson(client):
    r = client.post("/ask", json={"question": "What do agents use?", "stream": True})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "Answer citing [1]."


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
