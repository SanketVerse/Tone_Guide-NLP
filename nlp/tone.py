"""Rule-based linguistic tone / formality analysis.

This module is intentionally rule-based (not a black-box classifier)
so it is easy to explain in an NLP viva. It uses observable
linguistic features:

- contractions ("can't", "don't")
- slang / conversational expressions
- formal vocabulary & polite phrases
- academic vocabulary
- first/second-person pronoun usage
- exclamation marks, question marks, ALL-CAPS, emojis
- average sentence length
- emotional / humorous cue words

It scores each of the 8 target tones and returns the best one plus
per-tone scores for transparency.
"""

import re

TONES = [
    "Professional",
    "Formal",
    "Casual",
    "Friendly",
    "Emotional",
    "Academic",
    "Humorous",
    "Neutral",
]

CONTRACTIONS = [
    "can't", "don't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "wouldn't", "couldn't", "shouldn't",
    "mustn't", "i'm", "you're", "he's", "she's", "it's", "we're",
    "they're", "i've", "you've", "we've", "i'll", "you'll", "we'll",
    "gonna", "wanna", "gotta", "yeah", "hey", "hi",
]

SLANG = [
    "hey", "yeah", "yep", "nope", "cool", "awesome", "gonna", "wanna",
    "gotta", "kinda", "sorta", "stuff", "thing", "guys", "lol", "omg",
    "btw", "gimme", "lemme", "dunno", "yolo",
]

FORMAL_WORDS = [
    "regret", "inform", "furthermore", "moreover", "therefore", "hence",
    "consequently", "respectfully", "sincerely", "hereby", "herewith",
    "pursuant", "regarding", "concerning", "pleasure", "cordially",
    "apologize", "appreciate", "gratitude", "obliged",
]

POLITE_PHRASES = [
    "please", "thank you", "thanks", "kindly", "would you",
    "could you", "i would like", "i regret to inform",
]

PROFESSIONAL_WORDS = [
    "meeting", "schedule", "regards", "attached", "deadline", "project",
    "colleague", "commitment", "confirm", "update", "agenda", "priority",
    "deliverable", "action item", "follow up", "best regards",
]

ACADEMIC_WORDS = [
    "research", "analysis", "hypothesis", "methodology", "empirical",
    "significant", "phenomenon", "framework", "perspective", "furthermore",
    "notion", "discourse", "paradigm", "correlation", "variable",
    "assessment", "evaluation", "literature", "scholar", "thesis",
]

EMOTIONAL_WORDS = [
    "love", "hate", "heartbroken", "devastated", "thrilled", "ecstatic",
    "furious", "terrified", "cry", "tears", "miss you", "feel",
    "feeling", "sad", "happy", "angry", "sorry", "unfortunately",
    "amazing", "terrible", "wonderful", "awful", "hurt",
]

HUMOR_CUES = [
    "haha", "lol", "lmao", "hilarious", "funny", "joke", "joking",
    "kidding", "oops", "whoops", "plot twist", "literally dying",
    "can't even",
]

FIRST_PERSON = ["i", "me", "my", "mine", "we", "us", "our"]
SECOND_PERSON = ["you", "your", "yours"]


def _compile_phrase_pattern(phrases):
    """Compile phrases into an efficient single regex sorted by length descending."""
    clean = sorted({p.strip().lower() for p in phrases if p.strip()}, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(p) for p in clean) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# Pre-compile patterns once for zero per-call regex compilation overhead
_RE_CONTRACTIONS = _compile_phrase_pattern(CONTRACTIONS)
_RE_SLANG = _compile_phrase_pattern(SLANG)
_RE_FORMAL = _compile_phrase_pattern(FORMAL_WORDS)
_RE_POLITE = _compile_phrase_pattern(POLITE_PHRASES)
_RE_PROFESSIONAL = _compile_phrase_pattern(PROFESSIONAL_WORDS)
_RE_ACADEMIC = _compile_phrase_pattern(ACADEMIC_WORDS)
_RE_EMOTIONAL = _compile_phrase_pattern(EMOTIONAL_WORDS)
_RE_HUMOR = _compile_phrase_pattern(HUMOR_CUES)
_RE_FIRST_PERSON = _compile_phrase_pattern(FIRST_PERSON)
_RE_SECOND_PERSON = _compile_phrase_pattern(SECOND_PERSON)

_RE_CAPS = re.compile(r"\b[A-Z]{2,}\b")
_RE_WORDS = re.compile(r"\b\w+\b")
_RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_EMOJIS = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\u2600-\u27BF\u2B00-\u2BFF]+"
)


def extract_linguistic_features(text: str, preproc: dict = None) -> dict:
    """Extract interpretable linguistic features from raw text efficiently."""
    words = _RE_WORDS.findall(text)
    word_count = len(words)

    n_exclaim = text.count("!")
    n_question = text.count("?")
    n_ellipsis = text.count("...")
    caps_words = len(_RE_CAPS.findall(text))
    emojis = len(_RE_EMOJIS.findall(text))

    if preproc is None:
        sentences = [s for s in _RE_SENT_SPLIT.split(text.strip()) if s.strip()]
        sent_count = max(len(sentences), 1)
        avg_len = (word_count or 1) / sent_count
    else:
        avg_len = preproc.get("avg_sentence_length", 10)
        sent_count = preproc.get("sentence_count", 1)

    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_sentence_length": round(avg_len, 2),
        "contractions": len(_RE_CONTRACTIONS.findall(text)),
        "slang": len(_RE_SLANG.findall(text)),
        "formal_words": len(_RE_FORMAL.findall(text)),
        "polite_phrases": len(_RE_POLITE.findall(text)),
        "professional_words": len(_RE_PROFESSIONAL.findall(text)),
        "academic_words": len(_RE_ACADEMIC.findall(text)),
        "emotional_words": len(_RE_EMOTIONAL.findall(text)),
        "humor_cues": len(_RE_HUMOR.findall(text)),
        "first_person": len(_RE_FIRST_PERSON.findall(text)),
        "second_person": len(_RE_SECOND_PERSON.findall(text)),
        "exclamations": n_exclaim,
        "questions": n_question,
        "ellipsis": n_ellipsis,
        "caps_words": caps_words,
        "emojis": emojis,
    }


def analyze_tone(text: str, preproc: dict = None) -> dict:
    """Score 8 tones from linguistic features; return best + scores.

    Returns: {"tone": str, "confidence": float (0-100),
              "scores": {tone: float}, "features": {...}}
    Confidence = winner share of total score, scaled for readability.
    """
    if not text or not text.strip():
        return {
            "tone": "Neutral",
            "confidence": 0.0,
            "scores": {t: 0.0 for t in TONES},
            "features": {},
        }

    feats = extract_linguistic_features(text, preproc)
    wc = max(feats["word_count"], 1)
    avg = feats["avg_sentence_length"]

    scores = {t: 1.0 for t in TONES}  # small prior so Neutral is possible

    # --- Casual: contractions, slang, short sentences, exclamations ---
    scores["Casual"] += 3.0 * feats["contractions"] + 3.0 * feats["slang"]
    scores["Casual"] += 1.5 * feats["exclamations"] + 1.0 * feats["emojis"]
    scores["Casual"] += 1.5 * feats["second_person"]
    if avg < 10:
        scores["Casual"] += 2.0
    if feats["caps_words"]:
        scores["Casual"] += 1.0

    # --- Friendly: polite phrases, second person, exclamation, thanks ---
    scores["Friendly"] += 3.0 * feats["polite_phrases"]
    scores["Friendly"] += 1.5 * feats["second_person"]
    scores["Friendly"] += 1.5 * feats["first_person"]
    scores["Friendly"] += 1.5 * feats["exclamations"]
    if feats["emojis"]:
        scores["Friendly"] += 1.5

    # --- Professional: business vocab, polite, longer structured sentences ---
    scores["Professional"] += 3.0 * feats["professional_words"]
    scores["Professional"] += 1.5 * feats["polite_phrases"]
    scores["Professional"] += 1.0 * feats["formal_words"]
    if 12 <= avg <= 25:
        scores["Professional"] += 2.0
    if feats["contractions"] == 0 and wc > 4:
        scores["Professional"] += 1.5
    if feats["slang"] == 0 and feats["emojis"] == 0:
        scores["Professional"] += 1.0

    # --- Formal: formal words, no contractions/slang, long sentences ---
    scores["Formal"] += 3.5 * feats["formal_words"]
    scores["Formal"] += 1.5 * feats["polite_phrases"]
    if avg > 15:
        scores["Formal"] += 2.0
    if feats["contractions"] == 0 and wc > 4:
        scores["Formal"] += 2.0
    if feats["slang"] == 0 and feats["exclamations"] == 0:
        scores["Formal"] += 1.5

    # --- Academic: academic vocab, long sentences, no slang ---
    scores["Academic"] += 4.0 * feats["academic_words"]
    if avg > 18:
        scores["Academic"] += 2.5
    elif avg > 14:
        scores["Academic"] += 1.0
    if feats["slang"] == 0 and feats["contractions"] == 0 and wc > 6:
        scores["Academic"] += 1.5
    if feats["first_person"] == 0 and feats["second_person"] == 0 and wc > 8:
        scores["Academic"] += 1.0

    # --- Emotional: emotional words, exclamations, first person ---
    scores["Emotional"] += 3.0 * feats["emotional_words"]
    scores["Emotional"] += 1.5 * feats["exclamations"]
    scores["Emotional"] += 1.0 * feats["first_person"]
    scores["Emotional"] += 1.0 * feats["ellipsis"]
    if feats["caps_words"]:
        scores["Emotional"] += 1.0

    # --- Humorous: humor cues, exclamations, slang, questions ---
    scores["Humorous"] += 4.0 * feats["humor_cues"]
    scores["Humorous"] += 1.0 * feats["slang"]
    scores["Humorous"] += 1.0 * feats["exclamations"] + 0.5 * feats["questions"]

    # --- Neutral: short factual text with no strong markers ---
    markers = (
        feats["slang"]
        + feats["emotional_words"]
        + feats["humor_cues"]
        + feats["exclamations"]
        + feats["emojis"]
        + feats["caps_words"]
        + feats["formal_words"]
        + feats["academic_words"]
        + feats["professional_words"]
    )
    if markers == 0:
        scores["Neutral"] += 4.0
    if feats["questions"] == 0 and feats["exclamations"] == 0:
        scores["Neutral"] += 1.0

    total = sum(scores.values())
    best = max(scores, key=scores.get)
    share = scores[best] / total if total else 0
    confidence = round(40 + share * 58, 1)
    confidence = min(max(confidence, 0.0), 98.0)

    norm_scores = {k: round(v, 2) for k, v in scores.items()}
    return {
        "tone": best,
        "confidence": confidence,
        "scores": norm_scores,
        "features": feats,
    }
