"""Sentiment analysis using a pretrained Hugging Face model.

Model: distilbert-base-uncased-finetuned-sst-2-english
  - Outputs POSITIVE / NEGATIVE only, so low-confidence predictions
    (|score - 0.5| small) are mapped to Neutral. This mapping is
    documented and keeps the 3-class requirement (Positive / Negative
    / Neutral) without keyword rules.

Returns: {"sentiment": str, "confidence": float 0..1, "method": str}
"""

NEUTRAL_MARGIN = 0.15  # if |pos_score - 0.5| < margin -> Neutral


def _rule_fallback(text: str) -> dict:
    """Tiny VADER-style fallback if the HF model can't load.

    Uses NLTK VADER when available, else a minimal lexicon count.
    """
    lowered = text.lower()
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        compound = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        if compound >= 0.3:
            return {"sentiment": "Positive", "confidence": round(min(0.5 + compound / 2, 0.99), 4),
                    "method": "nltk-vader (fallback)"}
        if compound <= -0.3:
            return {"sentiment": "Negative", "confidence": round(min(0.5 - compound / 2, 0.99), 4),
                    "method": "nltk-vader (fallback)"}
        return {"sentiment": "Neutral", "confidence": round(1 - abs(compound), 4),
                "method": "nltk-vader (fallback)"}
    except Exception:
        pass
    pos_words = {"happy", "good", "great", "love", "excellent", "awesome",
                 "wonderful", "best", "glad", "pleased", "thank"}
    neg_words = {"sad", "bad", "hate", "terrible", "awful", "worst",
                 "disappointed", "angry", "sorry", "cannot", "can't",
                 "disappointing", "poor", "fail"}
    words = set(lowered.split())
    pos = len(words & pos_words)
    neg = len(words & neg_words)
    if pos > neg:
        return {"sentiment": "Positive", "confidence": 0.6,
                "method": "lexicon-count (fallback)"}
    if neg > pos:
        return {"sentiment": "Negative", "confidence": 0.6,
                "method": "lexicon-count (fallback)"}
    return {"sentiment": "Neutral", "confidence": 0.55,
            "method": "lexicon-count (fallback)"}


def analyze_sentiment(text: str) -> dict:
    """Classify text as Positive / Negative / Neutral with confidence."""
    if not text or not text.strip():
        return {"sentiment": "Neutral", "confidence": 0.0,
                "method": "empty-input"}
    # Keep input reasonably short for the model (truncate at 512 chars).
    snippet = text.strip()[:512]
    from models.model_loader import get_sentiment_pipeline
    pipe = get_sentiment_pipeline()
    if pipe is None:
        return _rule_fallback(text)
    try:
        result = pipe(snippet)[0]  # {"label": "POSITIVE", "score": 0.99}
        label = str(result["label"]).upper()
        score = float(result["score"])
        if label == "POSITIVE":
            pos_score = score
        else:  # NEGATIVE
            pos_score = 1.0 - score
        # Map low-confidence zone to Neutral.
        if abs(pos_score - 0.5) < NEUTRAL_MARGIN:
            return {"sentiment": "Neutral",
                    "confidence": round(1.0 - abs(pos_score - 0.5) * 2, 4),
                    "method": "distilbert-sst2 (neutral-mapped)"}
        if pos_score >= 0.5:
            return {"sentiment": "Positive", "confidence": round(pos_score, 4),
                    "method": "distilbert-sst2"}
        return {"sentiment": "Negative", "confidence": round(1.0 - pos_score, 4),
                "method": "distilbert-sst2"}
    except Exception:
        return _rule_fallback(text)
