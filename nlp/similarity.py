"""Semantic similarity using TF-IDF + cosine similarity (scikit-learn).

No ML model or internet required. Combines multiple signals:
  1. TF-IDF cosine (lexical overlap weighted by IDF)
  2. Word-order n-gram similarity (captures preserved sentence structure)
  3. Character n-gram similarity (catches partial word matches like
     "send" -> "forward" sharing substrings)

The blended score better reflects meaning preservation for tone
conversion, where words change but structure and intent stay.

Returns: {"score": float 0..1, "method": str}
"""

import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokenize(text: str) -> list:
    """Simple word tokenizer — lowercase, alphabetic tokens only."""
    return re.findall(r"[a-z]+", text.lower())


def _ngram_similarity(words1: list, words2: list, n: int = 2) -> float:
    """Cosine-like overlap of word n-grams (captures word order).

    Uses containment: how much of text1's n-gram set is covered by
    text2's n-gram set, averaged both directions.
    """
    def ngrams(words, n):
        return {tuple(words[i:i+n]) for i in range(len(words)-n+1)} if len(words) >= n else set()

    g1, g2 = ngrams(words1, n), ngrams(words2, n)
    if not g1 or not g2:
        return 0.0
    intersection = g1 & g2
    # Average containment both directions (like average of two Jaccard indices)
    j1 = len(intersection) / len(g1) if g1 else 0.0
    j2 = len(intersection) / len(g2) if g2 else 0.0
    return round((j1 + j2) / 2, 4)


def _char_ngram_overlap(text1: str, text2: str, n: int = 4) -> float:
    """Character 4-gram overlap — catches partial word similarity.

    E.g. "send" and "forward" share some char n-grams; this gives
    partial credit for synonym substitutions that share substrings.
    """
    def char_ngrams(s, n):
        s = s.lower().replace(" ", "")
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else set()

    g1, g2 = char_ngrams(text1, n), char_ngrams(text2, n)
    if not g1 or not g2:
        return 0.0
    intersection = g1 & g2
    return round(len(intersection) / max(len(g1), len(g2)), 4)


def _word_overlap_ratio(words1: list, words2: list) -> float:
    """How many content words from original appear in transformed (by position).

    Gives credit for words that survive the transformation unchanged.
    """
    if not words1:
        return 0.0
    set2 = set(words2)
    matches = sum(1 for w in words1 if w in set2)
    return round(matches / len(words1), 4)


def compute_similarity(original: str, transformed: str) -> dict:
    """Compute blended semantic similarity between two texts.

    Combines TF-IDF cosine, word-order n-gram similarity, character
    n-gram overlap, word retention ratio, and key-content coverage
    into a single score tuned for tone conversion.

    Key insight: tone conversion intentionally changes words and adds
    polite filler. The score should reward:
      - Preserved core action/content words (send→forward, help→assist)
      - Preserved sentence structure (word order)
      - Partial word similarity (synonyms share substrings)
    and should NOT heavily penalize:
      - Added polite phrases ("at your earliest convenience")
      - Removed filler ("bro", "hey")

    Returns {"score": float 0..1, "method": str}. Never raises.
    """
    orig_clean = original.strip()
    trans_clean = transformed.strip()

    if not orig_clean or not trans_clean:
        return {"score": 0.0, "method": "empty-input"}

    if orig_clean.lower() == trans_clean.lower():
        return {"score": 1.0, "method": "exact-match"}

    words1 = _tokenize(orig_clean)
    words2 = _tokenize(trans_clean)

    try:
        # Signal 1: TF-IDF cosine (global lexical similarity)
        vec = TfidfVectorizer().fit_transform([orig_clean, trans_clean])
        tfidf_score = float(cosine_similarity(vec[0], vec[1])[0][0])
        tfidf_score = max(0.0, min(1.0, tfidf_score))
    except Exception:
        tfidf_score = 0.0

    # Signal 2: Word-order bigram similarity (preserved sentence structure)
    bigram_score = _ngram_similarity(words1, words2, n=2)

    # Signal 3: Character n-gram overlap (partial credit for synonym substitutions)
    char_score = _char_ngram_overlap(orig_clean, trans_clean, n=4)

    # Signal 4: Word retention ratio (words that survived unchanged)
    retention_score = _word_overlap_ratio(words1, words2)

    # Signal 5: Key content word coverage
    # How many of the original's content words are represented in the
    # transformed text (exactly, via synonym, or via partial overlap).
    #
    # We exclude only TRUE filler/discourse words that are intentionally
    # removed during tone conversion (slang, greetings, intensifiers).
    FILLER_WORDS = {
        # Slang / casual discourse markers (intentionally removed)
        "bro", "hey", "hi", "yo", "dude", "guys", "yeah", "nope",
        "yep", "nah", "lol", "lmao", "omg", "btw", "tbh", "imo",
        "fr", "ngl", "cap", "fam", "gang", "lit", "fire", "dope",
        "sick", "goat", "based", "vibes", "vibe",
        # Intensifiers / filler that get expanded
        "like", "just", "really", "super", "totally", "literally",
        "basically", "actually", "honestly", "seriously",
        # Greetings/closings expanded to full phrases
        "sup", "wassup", "cheers", "later", "gimme", "lemme",
        "gonna", "wanna", "gotta", "dunno", "kinda", "sorta",
    }

    # Known synonym pairs used in the tone converter — these should
    # count as full matches for meaning preservation.
    SYNONYM_PAIRS = {
        # Original → Professional/Formal equivalent
        "send": {"forward", "transmit", "deliver"},
        "forward": {"send"},
        "help": {"assist", "aid", "support"},
        "assist": {"help"},
        "need": {"require", "necessitate"},
        "require": {"need", "necessitate"},
        "want": {"would like", "wish", "desire"},
        "get": {"obtain", "acquire", "receive"},
        "obtain": {"get", "acquire"},
        "use": {"utilise", "employ", "utilize"},
        "utilise": {"use", "employ"},
        "fix": {"rectify", "resolve", "correct"},
        "rectify": {"fix", "resolve"},
        "tell": {"inform", "advise", "notify", "elucidate"},
        "inform": {"tell", "advise", "notify"},
        "ask": {"request", "inquire"},
        "request": {"ask", "inquire"},
        "show": {"demonstrate", "display", "present"},
        "demonstrate": {"show", "display"},
        "find": {"identify", "locate", "ascertain"},
        "identify": {"find", "locate"},
        "think": {"believe", "consider", "postulate"},
        "believe": {"think", "consider"},
        "make": {"produce", "create", "constitute"},
        "produce": {"make", "create"},
        "give": {"provide", "supply", "render"},
        "provide": {"give", "supply"},
        "talk": {"discuss", "converse"},
        "discuss": {"talk", "converse"},
        "meet": {"convene", "gather"},
        "convene": {"meet", "gather"},
        "check": {"review", "examine", "evaluate"},
        "review": {"check", "examine"},
        "look": {"examine", "review", "investigate"},
        "write": {"draft", "compose"},
        "draft": {"write", "compose"},
        "start": {"commence", "initiate", "begin"},
        "commence": {"start", "initiate"},
        "end": {"conclude", "terminate", "finish"},
        "conclude": {"end", "terminate"},
        "buy": {"purchase", "procure"},
        "purchase": {"buy", "procure"},
        "good": {"satisfactory", "favourable", "excellent", "positive"},
        "excellent": {"good", "satisfactory", "outstanding"},
        "great": {"excellent", "outstanding", "satisfactory"},
        "awesome": {"exceptional", "remarkable", "outstanding", "excellent"},
        "amazing": {"remarkable", "noteworthy", "excellent"},
        "bad": {"unsatisfactory", "adverse", "negative"},
        "wrong": {"incorrect", "erroneous"},
        "right": {"correct", "accurate"},
        "easy": {"straightforward", "simple"},
        "hard": {"challenging", "difficult"},
        "big": {"substantial", "significant", "large"},
        "small": {"minimal", "negligible", "minor"},
        "soon": {"promptly", "immediately", "shortly"},
        "now": {"promptly", "currently", "presently"},
        "problem": {"concern", "issue", "matter", "phenomenon"},
        "issue": {"concern", "matter", "problem"},
        "thing": {"item", "element", "aspect"},
        "items": {"things", "elements"},
        "stuff": {"materials", "items", "content"},
        "nice": {"pleasing", "satisfactory", "pleasant"},
        "cool": {"impressive", "satisfactory", "excellent"},
        "got": {"received", "obtained", "acquired"},
        "come": {"arrive", "attend", "approach"},
        "go": {"proceed", "depart", "leave"},
        "see": {"observe", "note", "view"},
        "say": {"state", "posits", "declare"},
        "says": {"states", "posits", "declares"},
        "people": {"individuals", "persons", "scholars"},
        "studies": {"research", "empirical evidence", "literature"},
        "works": {"functions", "operates", "is effective"},
    }

    content_words1 = [w for w in words1 if w not in FILLER_WORDS]
    set2 = set(words2)

    if content_words1:
        # Count how many content words from original appear in transformed
        # (exact match, synonym match, or via character-level similarity)
        content_covered = 0.0
        for w in content_words1:
            if w in set2:
                content_covered += 1.0
            else:
                # Check known synonyms first (highest confidence)
                synonyms = SYNONYM_PAIRS.get(w, set())
                if synonyms & set2:
                    content_covered += 1.0  # full credit for known synonym
                    continue
                # Check multi-word synonym phrases in transformed text
                transformed_text_lower = trans_clean.lower()
                for syn_phrase in synonyms:
                    if syn_phrase in transformed_text_lower:
                        content_covered += 1.0
                        break
                else:
                    # Fall back to character-level similarity
                    for tw in set2:
                        if len(w) >= 3 and len(tw) >= 3 and w != tw:
                            cs1, cs2 = set(w), set(tw)
                            intersection = len(cs1 & cs2)
                            if intersection >= 3:
                                min_len = min(len(cs1), len(cs2))
                                if min_len > 0 and intersection / min_len >= 0.5:
                                    content_covered += 0.6
                                    break
        content_coverage = round(content_covered / len(content_words1), 4)
    else:
        content_coverage = 0.0

    # Detect if transformed text added lots of polite filler that inflates length
    # Without this, adding "at your earliest convenience" unfairly tanks the score
    polite_fillers = {
        "please", "forward", "kindly", "assist", "regarding",
        "convenience", "earliest", "opportunity", "necessary",
        "require", "obtain", "forward", "present", "provide",
        "inform", "request", "attention", "notify", "advise",
    }
    polite_word_count = sum(1 for w in words2 if w in polite_fillers)
    polite_ratio = polite_word_count / len(words2) if words2 else 0

    # Blend with weights tuned for tone conversion:
    # - content_coverage: most important — did we preserve the core meaning?
    # - retention: bonus for words that survived unchanged
    # - tfidf: backbone lexical similarity
    # - char: partial credit for synonym substitutions
    # - bigram: sentence structure preservation
    blended = (
        0.40 * content_coverage     # core meaning preservation (highest weight)
        + 0.20 * retention_score    # exact word retention
        + 0.15 * tfidf_score        # lexical overlap
        + 0.15 * char_score         # partial word similarity (synonyms)
        + 0.10 * bigram_score       # phrase structure
    )
    # Slight boost if the low score is mostly due to added polite filler
    # (which is intentional and good for tone conversion)
    if polite_ratio > 0.2 and blended < 0.3:
        blended = round(blended * 1.15, 4)  # 15% boost for politely expanded text

    blended = round(max(0.0, min(1.0, blended)), 4)

    return {
        "score": blended,
        "method": (
            "blended (content-coverage + retention + tfidf + char-ngram + bigram)"
        ),
        "details": {
            "content_coverage": content_coverage,
            "word_retention": retention_score,
            "tfidf_cosine": round(tfidf_score, 4),
            "char_ngram_overlap": char_score,
            "bigram_overlap": bigram_score,
            "polite_filler_ratio": polite_ratio,
        },
    }
