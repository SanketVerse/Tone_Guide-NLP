"""Centralised pretrained-model loading with thread-safe cached loaders and graceful fallbacks.

Models (all lightweight, laptop-friendly):
- Sentiment : distilbert-base-uncased-finetuned-sst-2-english
- Emotion   : j-hartmann/emotion-english-distilroberta-base
- Similarity: sentence-transformers/all-MiniLM-L6-v2 (lazy in similarity.py)
- Generation: google/flan-t5-small (instruction-tuned, CPU-friendly)
"""

from functools import lru_cache
import threading

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
GENERATION_MODEL = "google/flan-t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """Return HF sentiment pipeline or None (thread-safe)."""
    with _LOAD_LOCK:
        try:
            from transformers import pipeline
            return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
        except Exception as e:
            print(f"[ToneGuide] Sentiment model failed to load: {e}")
            return None


@lru_cache(maxsize=1)
def get_emotion_pipeline():
    """Return HF emotion pipeline or None (thread-safe)."""
    with _LOAD_LOCK:
        try:
            from transformers import pipeline
            return pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                top_k=None,  # return all label scores
            )
        except Exception as e:
            print(f"[ToneGuide] Emotion model failed to load: {e}")
            return None


@lru_cache(maxsize=1)
def get_generator():
    """Return HF text2text generator (FLAN-T5) or None (thread-safe)."""
    with _LOAD_LOCK:
        try:
            from transformers import pipeline
            return pipeline("text2text-generation", model=GENERATION_MODEL)
        except Exception as e:
            print(f"[ToneGuide] Generation model failed to load: {e}")
            return None


def clear_cache():
    """Clear cached models."""
    get_sentiment_pipeline.cache_clear()
    get_emotion_pipeline.cache_clear()
    get_generator.cache_clear()
