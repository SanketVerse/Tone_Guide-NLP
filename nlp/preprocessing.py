"""Text preprocessing for ToneGuide.

Keeps the ORIGINAL text unchanged for generation.
Preprocessing here is only for analysis (token stats, sentence
splitting, normalization) — we deliberately do NOT remove
stopwords or punctuation because that would hurt sentiment,
emotion and tone analysis.
"""

from functools import lru_cache
import re

try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


@lru_cache(maxsize=1)
def _ensure_punkt() -> bool:
    """Download NLTK punkt data once if missing (quiet, cached)."""
    if not _NLTK_AVAILABLE:
        return False
    for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            try:
                name = pkg.split("/")[-1]
                nltk.download(name, quiet=True)
            except Exception:
                pass
    return True


_UNICODE_MAP = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
})

_RE_SPACES = re.compile(r"[ \t]+")
_RE_MULTILINE = re.compile(r"\n\s*\n+")
_RE_NEWLINES = re.compile(r"\s*\n\s*")
_RE_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")
_RE_WORD_FALLBACK = re.compile(r"\w+|[^\w\s]")
_RE_WORD_CHAR = re.compile(r"\w")


def normalize_whitespace(text: str) -> str:
    """Collapse extra whitespace/newlines to single spaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_SPACES.sub(" ", text)
    text = _RE_MULTILINE.sub("\n", text)
    text = _RE_NEWLINES.sub(" ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Light normalization: unicode quotes/dashes + whitespace."""
    text = text.translate(_UNICODE_MAP)
    return normalize_whitespace(text)


def split_sentences(text: str) -> list:
    """Split text into sentences (NLTK if available, else regex)."""
    text = normalize_text(text)
    if not text:
        return []
    if _NLTK_AVAILABLE and _ensure_punkt():
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    parts = _RE_SENTENCE_REGEX.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> list:
    """Word-tokenize (NLTK if available, else regex)."""
    if _NLTK_AVAILABLE and _ensure_punkt():
        try:
            return _nltk_word_tokenize(text)
        except Exception:
            pass
    return _RE_WORD_FALLBACK.findall(text)


def preprocess_for_analysis(text: str) -> dict:
    """Run basic preprocessing and return analysis-ready info.

    Returns dict with: cleaned, sentences, tokens, word_count,
    sentence_count, avg_sentence_length, char_count.
    """
    cleaned = normalize_text(text)
    sentences = split_sentences(cleaned)
    tokens = tokenize_words(cleaned)
    words = [t for t in tokens if _RE_WORD_CHAR.search(t)]
    word_count = len(words)
    sentence_count = len(sentences) if sentences else (1 if cleaned else 0)
    avg_len = round(word_count / sentence_count, 2) if sentence_count else 0.0
    return {
        "cleaned": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_len,
        "char_count": len(cleaned),
    }
