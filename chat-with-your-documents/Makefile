PY ?= python

.PHONY: install ingest ui api chat eval test lint

install:
	$(PY) -m pip install -r requirements-dev.txt
ingest:
	$(PY) -m rag ingest
ui:
	streamlit run app.py
api:
	$(PY) -m rag serve
chat:
	$(PY) -m rag chat
eval:
	$(PY) -m rag eval
test:
	$(PY) -m pytest
lint:
	ruff check . && ruff format --check .
