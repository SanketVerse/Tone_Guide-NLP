"""Emotion detection using keyword/lexicon rules.

Uses a curated word list per emotion category. Scores are computed by
counting matching keywords; ties go to Neutral. All scores are
normalised so they sum to 1.0 and returned as `all_scores`.

Returns: {"emotion": str, "confidence": float 0..1,
          "all_scores": dict, "method": str}
"""

import re

# Emotion keyword lexicon
EMOTION_LEXICON = {
    "Joy": [
        "happy", "happiness", "glad", "great", "wonderful", "excited",
        "love", "awesome", "fantastic", "brilliant", "delighted",
        "cheerful", "pleased", "enjoy", "thrilled", "elated", "proud",
        "grateful", "joyful", "celebrate", "smile",
    ],
    "Sadness": [
        "sad", "disappointed", "unfortunately", "miss", "cry", "hurt",
        "grief", "sorrow", "upset", "heartbroken", "depressed",
        "lonely", "hopeless", "miserable", "regret", "mourn",
        "unhappy", "gloomy",
    ],
    "Anger": [
        "angry", "anger", "furious", "annoying", "hate", "frustrat",
        "irritat", "rage", "outraged", "mad", "hostile", "bitter",
        "resent", "infuriated", "livid",
    ],
    "Fear": [
        "afraid", "scared", "worried", "anxious", "fear", "nervous",
        "terror", "panic", "dread", "horrified", "frightened",
        "apprehensive", "uneasy", "phobia",
    ],
    "Surprise": [
        "wow", "surprised", "unexpected", "amazing", "shocking",
        "astonishing", "unbelievable", "incredible", "startled",
        "staggered", "stunned",
    ],
    "Disgust": [
        "disgusting", "gross", "repulsive", "sick of", "revolting",
        "nauseating", "offensive", "vile", "appalling",
    ],
}


def _count_hits(lowered: str, keywords: list) -> int:
    return sum(1 for w in keywords if w in lowered)


def analyze_emotion(text: str) -> dict:
    """Detect dominant emotion using keyword-based lexicon scoring."""
    if not text or not text.strip():
        return {"emotion": "Neutral", "confidence": 0.0,
                "all_scores": {}, "method": "empty-input"}

    lowered = text.strip().lower()
    raw_scores: dict[str, int] = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        raw_scores[emotion] = _count_hits(lowered, keywords)

    total_hits = sum(raw_scores.values())

    if total_hits == 0:
        # No keywords matched → Neutral
        all_scores = {e: 0.0 for e in EMOTION_LEXICON}
        all_scores["Neutral"] = 1.0
        return {"emotion": "Neutral", "confidence": 1.0,
                "all_scores": all_scores, "method": "keyword-lexicon"}

    # Normalise hit counts into probabilities
    all_scores = {e: round(c / total_hits, 4) for e, c in raw_scores.items()}
    best_emotion = max(raw_scores, key=raw_scores.get)
    confidence = round(raw_scores[best_emotion] / total_hits, 4)

    # Low dominance → Neutral
    if confidence < 0.35:
        all_scores["Neutral"] = round(1.0 - confidence, 4)
        return {"emotion": "Neutral", "confidence": round(1.0 - confidence, 4),
                "all_scores": all_scores, "method": "keyword-lexicon"}

    return {"emotion": best_emotion, "confidence": confidence,
            "all_scores": all_scores, "method": "keyword-lexicon"}
