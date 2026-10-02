"""Emotion detection using a pretrained Hugging Face model.

Model: j-hartmann/emotion-english-distilroberta-base
  Actual labels: anger, disgust, fear, joy, neutral, sadness, surprise.
  Display mapping (documented):
    anger    -> Anger        disgust -> Disgust
    fear     -> Fear          joy     -> Joy
    sadness  -> Sadness       surprise-> Surprise
    neutral  -> Neutral
  NOTE: this model has no "Love" label; that limitation is noted in
  the README. We display the model's real labels rather than forcing
  a Love category.

Returns: {"emotion": str, "confidence": float 0..1,
          "all_scores": dict, "method": str}
"""

LABEL_MAP = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Joy",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}


def _fallback(text: str) -> dict:
    """Keyword fallback if the HF model cannot load."""
    lowered = text.lower()
    cues = {
        "Joy": ["happy", "glad", "great", "wonderful", "excited", "love", "awesome"],
        "Sadness": ["sad", "disappointed", "sorry", "unfortunately", "miss", "cry", "hurt"],
        "Anger": ["angry", "furious", "annoying", "hate", "frustrat", "irritat"],
        "Fear": ["afraid", "scared", "worried", "anxious", "fear", "nervous"],
        "Surprise": ["wow", "surprised", "unexpected", "amazing", "shocking", "oh!"],
        "Disgust": ["disgusting", "gross", "repulsive", "sick of"],
    }
    best, best_hits = "Neutral", 0
    for emo, words in cues.items():
        hits = sum(1 for w in words if w in lowered)
        if hits > best_hits:
            best, best_hits = emo, hits
    conf = 0.55 if best == "Neutral" else min(0.6 + 0.1 * best_hits, 0.9)
    return {"emotion": best, "confidence": round(conf, 4),
            "all_scores": {best: conf}, "method": "keyword (fallback)"}


def analyze_emotion(text: str) -> dict:
    """Detect dominant emotion with confidence."""
    if not text or not text.strip():
        return {"emotion": "Neutral", "confidence": 0.0,
                "all_scores": {}, "method": "empty-input"}
    snippet = text.strip()[:512]
    from models.model_loader import get_emotion_pipeline
    pipe = get_emotion_pipeline()
    if pipe is None:
        return _fallback(text)
    try:
        raw = pipe(snippet)
        # pipeline(top_k=None) returns [[{label, score}, ...]]
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        scores = {}
        for item in raw:
            key = str(item["label"]).lower()
            scores[LABEL_MAP.get(key, key.title())] = round(float(item["score"]), 4)
        best = max(scores, key=scores.get)
        return {"emotion": best, "confidence": scores[best],
                "all_scores": scores,
                "method": "j-hartmann/emotion-english-distilroberta-base"}
    except Exception:
        return _fallback(text)
