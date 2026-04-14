"""Feature engineering for protein sequence classification."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def clean_sequence(sequence: str) -> str:
    """Keep only canonical amino-acid symbols."""
    allowed = set(AMINO_ACIDS)
    return "".join([ch for ch in sequence.upper() if ch in allowed])


def amino_acid_composition(sequence: str) -> list[float]:
    """Compute normalized amino-acid composition (20D vector)."""
    seq = clean_sequence(sequence)
    if not seq:
        return [0.0] * len(AMINO_ACIDS)
    length = len(seq)
    return [seq.count(aa) / length for aa in AMINO_ACIDS]


def build_aac_matrix(sequences: Iterable[str]) -> csr_matrix:
    """Build sparse matrix from amino-acid composition features."""
    rows = [amino_acid_composition(seq) for seq in sequences]
    return csr_matrix(np.asarray(rows, dtype=float))


def fit_tfidf_vectorizer(sequences: Iterable[str]) -> TfidfVectorizer:
    """Fit character-level TF-IDF vectorizer for sequence motifs."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    vectorizer.fit([clean_sequence(seq) for seq in sequences])
    return vectorizer


def build_feature_matrix(sequences: Iterable[str], vectorizer: TfidfVectorizer) -> csr_matrix:
    """Combine sequence n-gram TF-IDF and composition features."""
    cleaned = [clean_sequence(seq) for seq in sequences]
    tfidf = vectorizer.transform(cleaned)
    aac = build_aac_matrix(cleaned)
    return hstack([tfidf, aac], format="csr")
