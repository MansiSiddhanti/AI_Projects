FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding + reranker models into the image for offline startup.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY rag ./rag
COPY app.py ingest.py retrieve.py ./

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
