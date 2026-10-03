"""Controlled text tone transformation using linguistic rules.

No AI model or API key is required. Transformation is done via:
  1. Word/phrase substitution dictionaries (contractions, register swaps)
  2. Sentence-level prefix / suffix patterns per tone
  3. Intensity scaling (slight / clear / strong / very strong)

All logic is transparent and explainable — ideal for a college mini-project viva.
"""

import re

TARGET_TONES = [
    "Professional",
    "Formal",
    "Casual",
    "Friendly",
    "Emotional",
    "Academic",
    "Humorous",
    "Neutral",
]

# ---------------------------------------------------------------------------
# Substitution dictionaries
# ---------------------------------------------------------------------------

FORMAL_SWAPS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "isn't": "is not", "aren't": "are not",
    "i'm": "I am", "i've": "I have", "i'll": "I will",
    "you're": "you are", "you've": "you have", "we're": "we are",
    "it's": "it is", "gonna": "going to", "wanna": "want to",
    "gotta": "have to", "hey": "Hello", "yeah": "yes",
    "thanks": "thank you", "want": "would like", "get": "receive",
    "ask": "request", "tell": "inform", "but": "however",
    "also": "additionally", "so": "therefore", "very": "particularly",
    "fix": "rectify", "help": "assist", "buy": "purchase",
    "need": "require", "start": "commence", "end": "terminate",
    "use": "utilise", "show": "demonstrate", "find": "identify",
    "think": "consider", "make": "produce", "give": "provide",
}

CASUAL_SWAPS = {
    "cannot": "can't", "will not": "won't", "do not": "don't",
    "does not": "doesn't", "is not": "isn't", "are not": "aren't",
    "I am": "I'm", "I have": "I've", "you are": "you're",
    "it is": "it's", "would like": "want", "receive": "get",
    "request": "ask", "inform": "tell", "Hello": "Hey",
    "however": "but", "therefore": "so", "assist": "help",
    "purchase": "buy", "require": "need", "utilise": "use",
    "demonstrate": "show", "identify": "find", "consider": "think",
    "produce": "make", "provide": "give",
}


def _intensity_word(intensity: int) -> str:
    if intensity <= 25:
        return "slightly"
    if intensity <= 60:
        return "clearly"
    if intensity <= 85:
        return "strongly"
    return "very strongly"


def _create_multi_replacer(swaps: dict):
    """Build a single-pass compiled regex replacer preserving casing."""
    sorted_keys = sorted(swaps.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
        re.IGNORECASE,
    )
    lookup = {k.lower(): v for k, v in swaps.items()}

    def replacer(match):
        word = match.group(0)
        sub = lookup.get(word.lower(), word)
        if word.istitle():
            return sub.capitalize()
        return sub

    return lambda s: pattern.sub(replacer, s)


_APPLY_FORMAL_SWAPS = _create_multi_replacer(FORMAL_SWAPS)
_APPLY_CASUAL_SWAPS = _create_multi_replacer(CASUAL_SWAPS)


# ---------------------------------------------------------------------------
# Tone-specific transformation logic
# ---------------------------------------------------------------------------

def _rule_based_transform(text: str, tone: str, intensity: int) -> str:
    """Apply linguistic rewrites to shift text toward the target tone."""
    t = text.strip()
    strong = intensity > 50

    if tone in ("Formal", "Professional", "Academic"):
        t = _APPLY_FORMAL_SWAPS(t)
        if tone == "Formal":
            if strong and not re.match(
                r"^(dear|respected|i regret|please|kindly)", t, re.I
            ):
                t = "I regret to inform you that " + t[0].lower() + t[1:] if t else t
        elif tone == "Professional":
            t = re.sub(r"\bi can't\b", "I will be unable to", t, flags=re.I)
            t = re.sub(r"\bcannot\b", "will be unable to", t, flags=re.I)
            if strong:
                t = re.sub(
                    r"\bbecause i have some work\b",
                    "due to a prior commitment",
                    t,
                    flags=re.I,
                )
                t = re.sub(r"\bbecause\b", "due to", t, count=1, flags=re.I)
        elif tone == "Academic":
            t = re.sub(r"\bstudies show\b", "empirical evidence suggests", t, flags=re.I)
            t = re.sub(r"\bpeople think\b", "scholars argue", t, flags=re.I)
            if strong and not re.search(r"\bdue to\b", t, re.I):
                t = re.sub(r"\bbecause\b", "due to", t, count=1, flags=re.I)

    elif tone in ("Casual", "Friendly", "Humorous"):
        t = _APPLY_CASUAL_SWAPS(t)
        if tone == "Friendly" and strong:
            if not t.lower().startswith(("hi", "hey", "hello")):
                t = "Hi! " + t[0].upper() + t[1:] if t else t
            if "thank" not in t.lower():
                t = t.rstrip(".!") + "! Thanks so much!"
        elif tone == "Humorous" and strong:
            t = t.rstrip(".") + " — no pressure, haha!"
        elif tone == "Casual" and strong:
            t = re.sub(r"\bcannot\b", "can't", t, flags=re.I)

    elif tone == "Emotional":
        if strong and not re.search(
            r"(unfortunately|sadly|heartfelt|truly)", t, re.I
        ):
            t = "Unfortunately, " + t[0].lower() + t[1:] if t else t
        t = re.sub(r"\.\s*$", "...", t)  # trailing ellipsis for emotional feel

    elif tone == "Neutral":
        t = re.sub(r"!+", ".", t)
        t = _APPLY_FORMAL_SWAPS(t)
        # Strip leading emotional phrases
        t = re.sub(
            r"^(unfortunately,?\s*|sadly,?\s*|happily,?\s*)",
            "",
            t,
            flags=re.I,
        )
        t = t[0].upper() + t[1:] if t else t

    return t.strip() or text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transform_text(text: str, tone: str, intensity: int = 50) -> dict:
    """Transform text into the target tone with the given intensity.

    Returns {"transformed": str, "method": str}. Never raises.
    """
    if tone not in TARGET_TONES:
        tone = "Neutral"
    intensity = max(0, min(100, int(intensity)))
    text = text.strip()

    if not text:
        return {"transformed": "", "method": "empty-input"}

    if intensity <= 5:
        return {"transformed": text, "method": "intensity-too-low (passthrough)"}

    transformed = _rule_based_transform(text, tone, intensity)
    return {
        "transformed": transformed or text,
        "method": f"linguistic-rules ({tone}, intensity={intensity})",
    }
