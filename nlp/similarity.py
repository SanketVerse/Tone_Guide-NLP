"""Semantic similarity using TF-IDF + cosine similarity (scikit-learn).

No ML model or internet required. TF-IDF converts each text into a
sparse vector; cosine similarity measures how closely the two vectors
align in meaning.

Returns: {"score": float 0..1, "method": str}
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _jaccard_fallback(text1: str, text2: str) -> float:
    """Word-overlap Jaccard index — used only if sklearn fails."""
    a = set(text1.lower().split())
    b = set(text2.lower().split())
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def compute_similarity(original: str, transformed: str) -> dict:
    """Compute lexical similarity between two texts via TF-IDF cosine.

    Returns {"score": float 0..1, "method": str}. Never raises.
    """
    orig_clean = original.strip()
    trans_clean = transformed.strip()

    if not orig_clean or not trans_clean:
        return {"score": 0.0, "method": "empty-input"}

    if orig_clean.lower() == trans_clean.lower():
        return {"score": 1.0, "method": "exact-match"}

    try:
        vec = TfidfVectorizer().fit_transform([orig_clean, trans_clean])
        score = float(cosine_similarity(vec[0], vec[1])[0][0])
        score = round(max(0.0, min(1.0, score)), 4)
        return {"score": score, "method": "tfidf-cosine (sklearn)"}
    except Exception:
        return {
            "score": _jaccard_fallback(orig_clean, trans_clean),
            "method": "jaccard-overlap (fallback)",
        }
