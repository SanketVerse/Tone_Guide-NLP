"""Controlled text tone transformation.

Strategy (in priority order):
1. Optional external generation API (env vars) — for machines where a
   local generative model is too heavy.
2. Local lightweight instruction model: google/flan-t5-small via
   Hugging Face transformers (runs on CPU/MPS).
3. Linguistic rule-based fallback (transparent, explainable in viva)
   so the app always returns meaningful output without failure.
"""

import os
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

TONE_INSTRUCTIONS = {
    "Professional": "Rewrite the text in a concise professional workplace style suitable for an office email.",
    "Formal": "Rewrite the text in a very formal and polite style with no contractions, as in an official letter.",
    "Casual": "Rewrite the text in a relaxed casual conversational style, as if texting a friend.",
    "Friendly": "Rewrite the text in a warm friendly encouraging style with a positive tone.",
    "Emotional": "Rewrite the text in an expressive heartfelt emotional style that conveys feeling.",
    "Academic": "Rewrite the text in an objective academic scholarly style with precise vocabulary.",
    "Humorous": "Rewrite the text in a light-hearted humorous playful style while keeping the facts the same.",
    "Neutral": "Rewrite the text in a plain neutral factual style with no strong emotion.",
}


def _intensity_word(intensity: int) -> str:
    if intensity <= 25:
        return "slightly"
    if intensity <= 60:
        return "clearly"
    if intensity <= 85:
        return "strongly"
    return "very strongly"


def _build_prompt(text: str, tone: str, intensity: int) -> str:
    strength = _intensity_word(intensity)
    instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["Neutral"])
    return (
        f"{instruction} Apply the {tone} tone {strength} but preserve "
        f"the original meaning exactly and do not add new information.\n\n"
        f"Original: {text}\nRewritten:"
    )


# ---------------------------------------------------------------------------
# 1. External API path (optional)
# ---------------------------------------------------------------------------

def _external_api_available() -> bool:
    return bool(os.getenv("TONE_API_URL") and os.getenv("TONE_API_KEY"))


def _transform_via_api(text: str, tone: str, intensity: int) -> str:
    """Call an OpenAI-compatible chat-completions endpoint."""
    import requests

    url = os.getenv("TONE_API_URL", "").strip()
    key = os.getenv("TONE_API_KEY", "").strip()
    model = os.getenv("TONE_API_MODEL", "gpt-4o-mini").strip()
    strength = _intensity_word(intensity)
    system_msg = (
        f"You rewrite text in a {tone} tone ({strength}). "
        "Preserve the original meaning exactly. "
        "Do not add unrelated information. "
        "Reply with only the rewritten text."
    )
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3 + (intensity / 100) * 0.5,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 2. Local FLAN-T5 path
# ---------------------------------------------------------------------------

def _transform_via_local_model(text: str, tone: str, intensity: int) -> str:
    from models.model_loader import get_generator

    gen = get_generator()
    if gen is None:
        raise RuntimeError("local generation model unavailable")

    prompt = _build_prompt(text, tone, intensity)
    do_sample = intensity > 15
    gen_kwargs = {
        "max_new_tokens": 128,
        "do_sample": do_sample,
        "truncation": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = 0.3 + (intensity / 100) * 0.5

    out = gen(prompt, **gen_kwargs)
    result = out[0]["generated_text"].strip()
    for prefix in ("Rewritten:", "Output:", "Rewrite:"):
        if prefix in result:
            result = result.split(prefix)[-1].strip()
    if not result:
        raise RuntimeError("local model returned empty text")
    return result


# ---------------------------------------------------------------------------
# 3. Linguistic rule-based fallback (transparent, viva-friendly)
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
}

CASUAL_SWAPS = {
    "cannot": "can't", "will not": "won't", "do not": "don't",
    "does not": "doesn't", "is not": "isn't", "are not": "aren't",
    "I am": "I'm", "I have": "I've", "you are": "you're",
    "it is": "it's", "would like": "want", "receive": "get",
    "request": "ask", "inform": "tell", "Hello": "Hey",
    "however": "but", "therefore": "so", "assist": "help",
    "purchase": "buy", "require": "need",
}


def _create_multi_replacer(swaps: dict):
    """Build single-pass compiled regex replacer preserving casing."""
    sorted_keys = sorted(swaps.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
        re.IGNORECASE,
    )
    lookup = {k.lower(): v for k, v in swaps.items()}

    def replacer(match):
        word = match.group(0)
        lowered = word.lower()
        sub = lookup.get(lowered, word)
        if word.istitle():
            return sub.capitalize()
        return sub

    return lambda s: pattern.sub(replacer, s)


_APPLY_FORMAL_SWAPS = _create_multi_replacer(FORMAL_SWAPS)
_APPLY_CASUAL_SWAPS = _create_multi_replacer(CASUAL_SWAPS)


def _rule_based_transform(text: str, tone: str, intensity: int) -> str:
    """Linguistic rewrite with rich contextual substitutions."""
    t = text.strip()
    strong = intensity > 50

    if tone in ("Formal", "Professional", "Academic"):
        t = _APPLY_FORMAL_SWAPS(t)
        if tone == "Formal":
            if not re.match(r"^(dear|respected|i regret|please|kindly)", t, re.I):
                t = "I regret to inform you that " + t[0].lower() + t[1:] if t and strong else t
        elif tone == "Professional":
            t = re.sub(r"\bi can't\b", "I will be unable to", t, flags=re.I)
            t = re.sub(r"\bcannot\b", "will be unable to", t, flags=re.I)
            t = re.sub(r"\bbecause i have some work\b", "due to a prior commitment", t, flags=re.I)
            t = re.sub(r"\bbecause\b", "due to", t, count=1, flags=re.I) if strong else t
        elif tone == "Academic":
            if strong and not re.search(r"\bdue to\b", t, re.I):
                t = re.sub(r"\bbecause\b", "due to", t, count=1, flags=re.I)
    elif tone in ("Casual", "Friendly", "Humorous"):
        t = _APPLY_CASUAL_SWAPS(t)
        if tone == "Friendly" and strong:
            t = t.rstrip(".!") + "! Thanks so much!" if "thank" not in t.lower() else t
            if not t.lower().startswith(("hi", "hey", "hello")):
                t = "Hi! " + t[0].upper() + t[1:] if t else t
        elif tone == "Humorous" and strong:
            t = t.rstrip(".") + " — no pressure, haha!"
        elif tone == "Casual" and strong:
            t = re.sub(r"\bcannot\b", "can't", t, flags=re.I)
    elif tone == "Emotional":
        if strong and not re.search(r"(unfortunately|sadly|heartfelt|truly)", t, re.I):
            t = "Unfortunately, " + t[0].lower() + t[1:] if t else t
    elif tone == "Neutral":
        t = re.sub(r"!+", ".", t)
        t = _APPLY_FORMAL_SWAPS(t)

    return t.strip() or text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transform_text(text: str, tone: str, intensity: int = 50) -> dict:
    """Transform text into target tone with given intensity.

    Returns {"transformed": str, "method": str}.
    Never raises — falls back gracefully.
    """
    if tone not in TARGET_TONES:
        tone = "Neutral"
    intensity = max(0, min(100, int(intensity)))
    text = text.strip()
    if not text:
        return {"transformed": "", "method": "empty-input"}

    if intensity <= 5:
        return {"transformed": text, "method": "intensity-too-low (passthrough)"}

    # 1. External API if configured
    if _external_api_available():
        try:
            return {
                "transformed": _transform_via_api(text, tone, intensity),
                "method": "external-api (env-configured)",
            }
        except Exception as e:
            print(f"[ToneGuide] External API failed ({e}); trying local model.")

    # 2. Local FLAN-T5
    try:
        candidate = _transform_via_local_model(text, tone, intensity)
        if candidate.strip().lower() != text.strip().lower() and len(candidate.strip()) > 3:
            return {"transformed": candidate, "method": "google/flan-t5-small (local)"}
    except Exception as e:
        print(f"[ToneGuide] Local generation fallback ({e})")

    # 3. Rule-based fallback
    return {
        "transformed": _rule_based_transform(text, tone, intensity),
        "method": "linguistic-rules (fallback)",
    }
