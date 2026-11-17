"""Shared deterministic test doubles."""

import numpy as np


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeEmbeddingModel:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.seen_chunks = []

    def encode(self, chunks, normalize_embeddings=True, convert_to_numpy=True):
        self.seen_chunks.extend(chunks)
        vectors = []
        for chunk in chunks:
            lowered = chunk.lower()
            vector = np.array(
                [
                    lowered.count("python") + lowered.count("backend"),
                    lowered.count("react") + lowered.count("frontend"),
                    lowered.count("cloud") + lowered.count("docker"),
                    1.0,
                ],
                dtype=float,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


class UnavailableSemantic:
    def score(self, resume_text, job_text):
        from src.semantic import SemanticResult

        return SemanticResult(None, False, "Semantic model unavailable")

