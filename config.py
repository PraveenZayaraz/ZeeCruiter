"""Validated runtime configuration for ZeeCruiter."""

from dataclasses import dataclass
import os


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class AppConfig:
    max_pdf_size_mb: float = 10.0
    max_pdf_pages: int = 20
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_tokens: int = 220
    semantic_chunk_overlap: int = 32
    semantic_max_chunks: int = 24
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral"
    ollama_timeout_seconds: int = 20
    weight_semantic: float = 0.50
    weight_lexical: float = 0.20
    weight_skills: float = 0.30

    def __post_init__(self) -> None:
        weights = (self.weight_semantic, self.weight_lexical, self.weight_skills)
        if any(weight < 0 for weight in weights):
            raise ValueError("Scoring weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Scoring weights must sum to 1.0")
        if self.max_pdf_size_mb <= 0 or self.max_pdf_pages <= 0:
            raise ValueError("PDF limits must be positive")
        if self.semantic_chunk_tokens < 8:
            raise ValueError("SEMANTIC_CHUNK_TOKENS must be at least 8")
        if not 0 <= self.semantic_chunk_overlap < self.semantic_chunk_tokens:
            raise ValueError("Semantic chunk overlap must be smaller than chunk size")
        if self.semantic_max_chunks <= 0:
            raise ValueError("SEMANTIC_MAX_CHUNKS must be positive")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be positive")

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            max_pdf_size_mb=_env_float("MAX_PDF_SIZE_MB", 10.0),
            max_pdf_pages=_env_int("MAX_PDF_PAGES", 20),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            semantic_chunk_tokens=_env_int("SEMANTIC_CHUNK_TOKENS", 220),
            semantic_chunk_overlap=_env_int("SEMANTIC_CHUNK_OVERLAP", 32),
            semantic_max_chunks=_env_int("SEMANTIC_MAX_CHUNKS", 24),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
            ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 20),
            weight_semantic=_env_float("WEIGHT_SEMANTIC", 0.50),
            weight_lexical=_env_float("WEIGHT_LEXICAL", 0.20),
            weight_skills=_env_float("WEIGHT_SKILLS", 0.30),
        )


CONFIG = AppConfig.from_env()

