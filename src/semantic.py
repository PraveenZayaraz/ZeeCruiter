"""Lazy MiniLM scoring with token-aware chunked document embeddings."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import CONFIG, AppConfig


@dataclass(frozen=True)
class SemanticResult:
    score: float | None
    available: bool
    message: str
    resume_chunks: int = 0
    job_chunks: int = 0
    resume_truncated: bool = False
    job_truncated: bool = False


class SemanticScorer:
    def __init__(self, config: AppConfig = CONFIG, model: Any = None):
        self.config = config
        self._model = model
        self._load_error: str | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model_name)
            return self._model
        except Exception as error:
            self._load_error = error.__class__.__name__
            return None

    def _chunk(self, text: str, model: Any) -> tuple[list[str], bool]:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            words = text.split()
            token_ids: list[Any] = words
            decode = lambda values: " ".join(values)
        else:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decode = lambda values: tokenizer.decode(values, skip_special_tokens=True)

        size = self.config.semantic_chunk_tokens
        overlap = self.config.semantic_chunk_overlap
        step = size - overlap
        chunks: list[str] = []
        truncated = False
        for start in range(0, len(token_ids), step):
            piece = token_ids[start : start + size]
            if not piece:
                break
            decoded = decode(piece).strip()
            if decoded:
                chunks.append(decoded)
            if len(chunks) >= self.config.semantic_max_chunks:
                truncated = start + size < len(token_ids)
                break
            if start + size >= len(token_ids):
                break
        return chunks, truncated

    def _document_embedding(self, text: str, model: Any) -> tuple[np.ndarray, int, bool]:
        chunks, truncated = self._chunk(text, model)
        if not chunks:
            raise ValueError("Cannot embed an empty document")
        embeddings = np.asarray(
            model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True),
            dtype=float,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        document_embedding = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(document_embedding))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The embedding model returned an invalid vector")
        return document_embedding / norm, len(chunks), truncated

    def score(self, resume_text: str, job_text: str) -> SemanticResult:
        if not resume_text.strip() or not job_text.strip():
            return SemanticResult(None, False, "Both documents require text")
        model = self._load_model()
        if model is None:
            return SemanticResult(
                None,
                False,
                "MiniLM is unavailable; install/download the configured model to enable semantic scoring.",
            )
        try:
            resume_embedding, resume_chunks, resume_truncated = self._document_embedding(resume_text, model)
            job_embedding, job_chunks, job_truncated = self._document_embedding(job_text, model)
            similarity = float(np.dot(resume_embedding, job_embedding))
            return SemanticResult(
                float(np.clip(similarity, 0.0, 1.0)),
                True,
                "MiniLM semantic scoring enabled",
                resume_chunks,
                job_chunks,
                resume_truncated,
                job_truncated,
            )
        except Exception as error:
            return SemanticResult(
                None,
                False,
                f"Semantic scoring failed safely ({error.__class__.__name__}).",
            )
