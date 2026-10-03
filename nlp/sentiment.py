"""Sentiment analysis using NLTK VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pure
lexicon-and-rule-based tool included in NLTK. It requires no ML model
and no internet connection after first download.

Returns: {"sentiment": str, "confidence": float 0..1, "method": str}
"""

import nltk


def _ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def _lexicon_fallback(text: str) -> dict:
    """Minimal keyword-count fallback if VADER is unavailable."""
    lowered = text.lower()
    pos_words = {
        "happy", "good", "great", "love", "excellent", "awesome",
        "wonderful", "best", "glad", "pleased", "thank", "enjoy",
        "nice", "fantastic", "brilliant", "superb", "positive",
    }
    neg_words = {
        "sad", "bad", "hate", "terrible", "awful", "worst",
        "disappointed", "angry", "sorry", "cannot", "can't",
        "disappointing", "poor", "fail", "horrible", "dreadful",
    }
    words = set(lowered.split())
    pos = len(words & pos_words)
    neg = len(words & neg_words)
    if pos > neg:
        return {"sentiment": "Positive", "confidence": 0.60,
                "method": "lexicon-count"}
    if neg > pos:
        return {"sentiment": "Negative", "confidence": 0.60,
                "method": "lexicon-count"}
    return {"sentiment": "Neutral", "confidence": 0.55,
            "method": "lexicon-count"}


def analyze_sentiment(text: str) -> dict:
    """Classify text as Positive / Negative / Neutral using NLTK VADER."""
    if not text or not text.strip():
        return {"sentiment": "Neutral", "confidence": 0.0,
                "method": "empty-input"}
    try:
        _ensure_vader()
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text.strip())
        compound = scores["compound"]
        # compound is in [-1, 1]; convert to confidence in [0, 1]
        if compound >= 0.05:
            confidence = round(min(0.5 + compound / 2, 0.99), 4)
            return {"sentiment": "Positive", "confidence": confidence,
                    "method": "nltk-vader"}
        if compound <= -0.05:
            confidence = round(min(0.5 + abs(compound) / 2, 0.99), 4)
            return {"sentiment": "Negative", "confidence": confidence,
                    "method": "nltk-vader"}
        # Near-zero compound → Neutral
        confidence = round(1.0 - abs(compound), 4)
        return {"sentiment": "Neutral", "confidence": confidence,
                "method": "nltk-vader"}
    except Exception:
        return _lexicon_fallback(text)
