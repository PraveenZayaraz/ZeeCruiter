"""Stable lexical similarity using a stateless hashing vectorizer."""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class LexicalScorer:
    """Compute a role-independent word/phrase cosine similarity.

    Unlike pair-fitted TF-IDF, the feature space does not change when another role is
    added to a comparison. This keeps a job's lexical score stable across batches.
    """

    def __init__(self, n_features: int = 2**16):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            binary=True,
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def score(self, resume_text: str, job_text: str) -> float | None:
        if not resume_text.strip() or not job_text.strip():
            return None
        matrix = self.vectorizer.transform([resume_text, job_text])
        similarity = float(matrix[0].multiply(matrix[1]).sum())
        return float(np.clip(similarity, 0.0, 1.0))


