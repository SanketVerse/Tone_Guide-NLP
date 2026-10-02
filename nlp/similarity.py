"""Semantic similarity via Sentence Transformers + cosine similarity.

Maps both texts to dense sentence embeddings, then computes cosine
similarity in [0, 1]. A higher score means the two texts are closer
in meaning (embedding space).
"""

from functools import lru_cache


def _cosine_fallback(text1: str, text2: str) -> float:
    """Fallback when sentence-transformers is unavailable.

    Uses TF-IDF + cosine similarity from scikit-learn.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not text1.strip() or not text2.strip():
            return 0.0
        vec = TfidfVectorizer().fit_transform([text1, text2])
        score = float(cosine_similarity(vec[0], vec[1])[0][0])
        return round(max(0.0, min(1.0, score)), 4)
    except Exception:
        a = set(text1.lower().split())
        b = set(text2.lower().split())
        if not a or not b:
            return 0.0
        return round(len(a & b) / len(a | b), 4)


@lru_cache(maxsize=1)
def _load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    # Using cpu for tiny sentence pairs is 50x faster on Mac than MPS kernel compile
    return SentenceTransformer(model_name, device="cpu")


def compute_similarity(original: str, transformed: str) -> dict:
    """Compute semantic similarity between two texts.

    Returns {"score": float 0..1, "method": str}.
    Never raises — falls back gracefully.
    """
    orig_clean = original.strip()
    trans_clean = transformed.strip()

    if not orig_clean or not trans_clean:
        return {"score": 0.0, "method": "empty-input"}

    # Fast check for identical text
    if orig_clean.lower() == trans_clean.lower():
        return {"score": 1.0, "method": "exact-match"}

    try:
        model = _load_embedder()
        embeddings = model.encode(
            [orig_clean, trans_clean],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        from sentence_transformers.util import cos_sim
        score = float(cos_sim(embeddings[0], embeddings[1]).item())
        score = max(0.0, min(1.0, score))
        return {
            "score": round(score, 4),
            "method": "sentence-transformers/all-MiniLM-L6-v2 + cosine",
        }
    except Exception:
        return {
            "score": _cosine_fallback(orig_clean, trans_clean),
            "method": "tfidf-cosine (fallback)",
        }
