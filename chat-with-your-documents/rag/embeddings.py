from typing import Protocol


class Embedder(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        # counting tokens on long text spams a max-length warning
        hf_logging.set_verbosity_error()

        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.max_tokens = self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
