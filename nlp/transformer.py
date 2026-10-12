"""Controlled text tone transformation using enhanced linguistic rules.

No AI model or API key is required. Transformation is done via:
  1. Casual/slang normalisation (expanded vocabulary)
  2. Word/phrase substitution dictionaries per tone
  3. Sentence-level structural rewriting per tone
  4. Prefix / suffix patterns per tone
  5. Intensity scaling

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
# Step 1 — Normalise casual/slang BEFORE tone substitution
# ---------------------------------------------------------------------------

SLANG_NORMALIZE = {
    # Vocative address words — strip them (they have no place in formal text)
    "bro": "",
    "dude": "",
    "buddy": "",
    "mate": "",
    "pal": "",
    "guys": "everyone",
    "y'all": "everyone",
    "man": "",
    "folk": "people",
    "fam": "",
    "gang": "",
    "team": "",
    # Intensifiers / filler
    "asap": "as soon as possible",
    "ASAP": "as soon as possible",
    "tbh": "to be honest",
    "ngl": "honestly",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "btw": "by the way",
    "fyi": "for your information",
    "omg": "oh my",
    "lol": "",
    "lmao": "",
    "lolz": "",
    "haha": "",
    "hehe": "",
    "hihi": "",
    "kinda": "somewhat",
    "sorta": "somewhat",
    "kinda like": "similar to",
    "basically": "essentially",
    "literally": "",
    "like": "",         # discourse filler "like" is hard; strip bare instances
    "super": "very",
    "totally": "completely",
    "legit": "genuinely",
    "fr": "truly",
    "lowkey": "somewhat",
    "highkey": "definitely",
    "no cap": "honestly",
    "for real": "genuinely",
    "chill": "relax",
    "chilling": "relaxing",
    "chillin": "relaxing",
    "vibes": "atmosphere",
    "vibe": "feel",
    "lit": "excellent",
    "fire": "excellent",
    "sick": "excellent",
    "dope": "excellent",
    "goat": "exceptional",
    "based": "admirable",
    # Actions / verbs
    "gimme": "give me",
    "lemme": "let me",
    "gotta": "need to",
    "gonna": "going to",
    "wanna": "want to",
    "dunno": "do not know",
    "hafta": "have to",
    "betcha": "I bet",
    "imo": "in my opinion",
    "gimme": "provide me with",
    # Greetings / closings
    "hey": "Hello",
    "yo": "Hello",
    "sup": "How are you",
    "wassup": "How are you",
    "hiya": "Hello",
    "cheers": "thank you",
    "later": "goodbye",
    "catch you": "goodbye",
    # Negation / response
    "nope": "no",
    "yep": "yes",
    "yeah": "yes",
    "yup": "yes",
    "nah": "no",
    "feels": "feels",
    # Contractions (informal → spelled out)
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "i'm": "I am",
    "i've": "I have",
    "i'll": "I will",
    "i'd": "I would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "he's": "he is",
    "she's": "she is",
    "who's": "who is",
    "what's": "what is",
}

# ---------------------------------------------------------------------------
# Step 2 — Tone-specific vocabulary swaps (applied AFTER normalisation)
# ---------------------------------------------------------------------------

PROFESSIONAL_SWAPS = {
    "send me": "please forward",
    "send": "forward",
    "slides": "the presentation slides",
    "the slides": "the presentation slides",
    "get": "obtain",
    "need": "require",
    "want": "would appreciate",
    "use": "utilise",
    "fix": "rectify",
    "tell": "advise",
    "ask": "kindly request",
    "help": "assist",
    "buy": "purchase",
    "start": "commence",
    "end": "conclude",
    "show": "demonstrate",
    "find": "identify",
    "think": "believe",
    "make": "produce",
    "give": "provide",
    "talk": "discuss",
    "meet": "convene",
    "check": "review",
    "look at": "review",
    "look into": "investigate",
    "call": "contact",
    "write": "draft",
    "set up": "arrange",
    "follow up": "follow up on",
    "because": "due to the fact that",
    "but": "however",
    "also": "additionally",
    "very": "particularly",
    "a lot of": "a significant amount of",
    "lots of": "numerous",
    "big": "substantial",
    "small": "minimal",
    "soon": "at the earliest convenience",
    "now": "promptly",
    "later": "at a later time",
    "maybe": "it may be possible that",
    "okay": "acknowledged",
    "ok": "acknowledged",
    "good": "satisfactory",
    "great": "excellent",
    "awesome": "exceptional",
    "amazing": "remarkable",
    "cool": "satisfactory",
    "nice": "pleasing",
    "bad": "unsatisfactory",
    "wrong": "incorrect",
    "right": "correct",
    "easy": "straightforward",
    "hard": "challenging",
    "problem": "concern",
    "issue": "matter",
    "thing": "item",
    "awesome": "exceptional",
    "cool": "impressive",
    "stuff": "materials",
    "things": "items",
}

FORMAL_SWAPS = {
    **PROFESSIONAL_SWAPS,
    "please": "I kindly request that you",
    "thank you": "I express my sincere gratitude",
    "sorry": "I sincerely apologise",
    "I am sorry": "I sincerely apologise",
    "I would like": "I wish to",
    "I want": "I wish to",
    "get back to": "revert to",
    "get back": "respond",
    "let me know": "kindly inform me",
    "I think": "It is my considered opinion that",
    "I believe": "It is my considered opinion that",
    "contact": "correspond with",
    "meet": "convene a meeting",
    "tell": "hereby inform",
    "need": "necessitate",
}

CASUAL_SWAPS = {
    # Formal → Casual
    "cannot": "can't",
    "will not": "won't",
    "do not": "don't",
    "does not": "doesn't",
    "is not": "isn't",
    "are not": "aren't",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
    "I would": "I'd",
    "you are": "you're",
    "you have": "you've",
    "we are": "we're",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "would like": "want",
    "require": "need",
    "obtain": "get",
    "receive": "get",
    "request": "ask",
    "inform": "tell",
    "assist": "help",
    "purchase": "buy",
    "utilise": "use",
    "utilise": "use",
    "demonstrate": "show",
    "identify": "find",
    "consider": "think about",
    "provide": "give",
    "however": "but",
    "therefore": "so",
    "additionally": "also",
    "furthermore": "also",
    "regarding": "about",
    "Hello": "Hey",
    "Greetings": "Hey",
    "acknowledged": "okay",
    "satisfactory": "good",
    "unsatisfactory": "not great",
    "immediately": "right now",
    "straightforward": "easy",
    "challenging": "tough",
    "concern": "problem",
    "matter": "thing",
    "forward": "send",
    "draft": "write",
    "convene": "meet",
    "investigate": "look into",
    "review": "check",
    "commence": "start",
    "conclude": "end",
    "substantial": "big",
    "minimal": "small",
}

FRIENDLY_SWAPS = {
    **CASUAL_SWAPS,
    "do not": "don't",
    "cannot": "can't",
    "Hello": "Hi",
    "greetings": "Hi there",
    "problem": "issue",
    "immediately": "soon",
    "require": "need",
    "obtain": "get",
    "correct": "right",
    "incorrect": "wrong",
}

ACADEMIC_SWAPS = {
    "says": "posits",
    "shows": "demonstrates",
    "proves": "substantiates",
    "thinks": "hypothesises",
    "thinks that": "postulates that",
    "people think": "scholars argue",
    "studies show": "empirical evidence suggests",
    "many people": "a considerable proportion of individuals",
    "a lot of": "a substantial number of",
    "lots of": "numerous",
    "because": "owing to the fact that",
    "so": "consequently",
    "but": "however",
    "also": "furthermore",
    "and": "and additionally",
    "use": "employ",
    "find": "ascertain",
    "look at": "examine",
    "look into": "investigate",
    "think about": "contemplate",
    "help": "facilitate",
    "big": "significant",
    "small": "negligible",
    "very": "considerably",
    "a lot": "extensively",
    "I think": "It is posited that",
    "I believe": "One may argue that",
    "we can see": "it can be observed that",
    "problem": "phenomenon",
    "issue": "concern",
    "shows that": "demonstrates that",
    "result": "outcome",
    "make": "constitute",
    "get": "acquire",
    "need": "necessitate",
    "start": "initiate",
    "end": "conclude",
    "check": "evaluate",
    "tell": "elucidate",
    "explain": "elaborate on",
    "good": "favourable",
    "bad": "adverse",
    "awesome": "remarkable",
    "amazing": "noteworthy",
    "great": "substantial",
    "important": "significant",
    "different": "disparate",
    "same": "identical",
}

# ---------------------------------------------------------------------------
# Helper: build fast compiled replacer
# ---------------------------------------------------------------------------

def _create_multi_replacer(swaps: dict):
    """Build a single-pass compiled regex replacer preserving sentence casing."""
    sorted_keys = sorted(swaps.keys(), key=len, reverse=True)
    if not sorted_keys:
        return lambda s: s
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
        re.IGNORECASE,
    )
    lookup = {k.lower(): v for k, v in swaps.items()}

    def replacer(match):
        word = match.group(0)
        sub = lookup.get(word.lower(), word)
        if not sub:          # empty → remove token + trailing space handled below
            return ""
        if word[0].isupper() and word.lower() != word:
            return sub[0].upper() + sub[1:]
        return sub

    def apply(s):
        result = pattern.sub(replacer, s)
        # Collapse multiple spaces that may arise from empty substitutions
        result = re.sub(r" {2,}", " ", result).strip()
        return result

    return apply


_NORMALIZE_SLANG = _create_multi_replacer(SLANG_NORMALIZE)
_APPLY_PROFESSIONAL = _create_multi_replacer(PROFESSIONAL_SWAPS)
_APPLY_FORMAL = _create_multi_replacer(FORMAL_SWAPS)
_APPLY_CASUAL = _create_multi_replacer(CASUAL_SWAPS)
_APPLY_FRIENDLY = _create_multi_replacer(FRIENDLY_SWAPS)
_APPLY_ACADEMIC = _create_multi_replacer(ACADEMIC_SWAPS)


# ---------------------------------------------------------------------------
# Step 3 — Sentence-level structural rewriting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _capitalise_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _ensure_period(s: str) -> str:
    if s and s[-1] not in ".!?":
        return s + "."
    return s


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip(".!?,;:")


def _cleanup_punctuation_artifacts(s: str) -> str:
    """Clean up punctuation artifacts left after word removal (e.g., 'awesome ,' → 'awesome,')."""
    # Fix space before punctuation
    s = re.sub(r'\s+([,!?;:])', r'\1', s)
    # Fix double punctuation (including mixed like !. or ?.)
    s = re.sub(r'[,]{2,}', ',', s)
    s = re.sub(r'[!]{2,}', '!', s)
    s = re.sub(r'[?]{2,}', '?', s)
    # Fix mixed double punctuation like !. or ?. → just .
    s = re.sub(r'[!?]\.', '.', s)
    s = re.sub(r'\.[!?]', '.', s)
    # Remove standalone punctuation at start
    s = re.sub(r'^[,!?;\s]+', '', s)
    # Remove trailing comma/semicolon before period
    s = re.sub(r',\s*\.', '.', s)
    s = re.sub(r';\s*\.', '.', s)
    return s.strip()


def _fix_grammar_after_substitution(s: str) -> str:
    """Fix common grammar issues after word substitution."""
    # "require to" → rephrase
    s = re.sub(r'\brequire\s+to\b', 'need to', s, flags=re.I)
    # "necessitate to" → "necessitate"
    s = re.sub(r'\bnecessitate\s+to\b', 'necessitate', s, flags=re.I)
    # "I require this done" → "I require this to be done"
    s = re.sub(r'\bi\s+require\s+this\s+done\b', 'I require this to be completed', s, flags=re.I)
    s = re.sub(r'\bi\s+require\s+that\s+done\b', 'I require that this be completed', s, flags=re.I)
    # "need this done" → "need this to be done"
    s = re.sub(r'\bneed\s+this\s+done\b', 'need this to be completed', s, flags=re.I)
    s = re.sub(r'\brequire\s+this\s+done\b', 'require this to be completed', s, flags=re.I)
    # "check this out" → "review this"
    s = re.sub(r'\bcheck\s+this\s+out\b', 'review this', s, flags=re.I)
    # "look at this" → "review this"
    s = re.sub(r'\blook\s+at\s+this\b', 'review this', s, flags=re.I)
    # "review this out" → "review this" (from check+out then review)
    s = re.sub(r'\breview\s+this\s+out\b', 'review this', s, flags=re.I)
    # Deduplicate polite markers from stacked substitutions (e.g. "can you please please forward")
    s = re.sub(r'\bplease\s+please\b', 'please', s, flags=re.I)
    s = re.sub(r'\bkindly\s+kindly\b', 'kindly', s, flags=re.I)
    s = re.sub(r'\bplease\s+kindly\b', 'kindly', s, flags=re.I)
    s = re.sub(r'\bkindly\s+please\b', 'kindly', s, flags=re.I)
    # Capitalize standalone "i" (but not inside words)
    s = re.sub(r'\bi\b', 'I', s)
    return s


def _rewrite_sentence_professional(sent: str, intensity: int) -> str:
    """Rewrite a single sentence to professional register."""
    s = sent.strip()
    # Strip greeting artefacts that may survive after slang normalisation
    s = re.sub(r'^(Hello|Hi|Hey)[,!]?\s*', '', s, flags=re.I).strip()
    s = _capitalise_first(s)

    # Convert "send me X ASAP" → "please forward X at your earliest convenience"
    s = re.sub(r'\bsend\s+me\b', 'please forward', s, flags=re.I)
    s = re.sub(r'\bASAP\b', 'at your earliest convenience', s)
    s = re.sub(r'\bas soon as possible\b', 'at your earliest convenience', s, flags=re.I)

    # "I cannot / I can't come to the meeting" → proper professional phrasing
    s = re.sub(r"\bi(?:'m| am)\s+unable\s+to\b", "I will be unable to", s, flags=re.I)
    s = re.sub(r"\bcannot\s+come\s+to\b", "will be unable to attend", s, flags=re.I)
    s = re.sub(r"\bcannot\s+come\b", "will be unable to attend the", s, flags=re.I)
    s = re.sub(r"\bcannot\s+attend\b", "will be unable to attend", s, flags=re.I)
    s = re.sub(r"\bdo not\s+come\b", "will not be attending", s, flags=re.I)

    # "because I have some work" → "due to a prior commitment"
    s = re.sub(r'\bbecause\s+i\s+have\s+some\s+work\b', 'due to a prior commitment', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\s+am\s+busy\b', 'due to a prior commitment', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\s+have\s+a\b', 'owing to a', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\b', 'owing to the fact that I', s, flags=re.I)
    s = re.sub(r'\bbecause\b', 'due to the fact that', s, flags=re.I)

    # Apply vocabulary swaps
    s = _APPLY_PROFESSIONAL(s)

    # Fix grammar issues from substitutions
    s = _fix_grammar_after_substitution(s)

    # Clean up any punctuation artifacts from word removal
    s = _cleanup_punctuation_artifacts(s)

    # High-intensity: add polite opening if imperative/request
    if intensity > 60:
        # Detect imperative-style requests (starts with verb)
        if re.match(r'^(please\s+)?(send|share|forward|give|tell|call|write|check|review|submit|complete|update|provide)\b', s, re.I):
            if not s.lower().startswith("please"):
                s = "Please " + s[0].lower() + s[1:]

    # Ensure professional ending (replace ! with . for professional tone)
    s = re.sub(r'!+', '.', s)
    
    return _capitalise_first(_ensure_period(s))


def _rewrite_sentence_formal(sent: str, intensity: int) -> str:
    s = sent.strip()
    # Strip greeting artefacts
    s = re.sub(r'^(Hello|Hi|Hey)[,!]?\s*', '', s, flags=re.I).strip()
    s = _capitalise_first(s)
    s = re.sub(r'\bASAP\b', 'at the earliest opportunity', s)
    s = re.sub(r'\bas soon as possible\b', 'at the earliest opportunity', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\s+have\s+some\s+work\b', 'on account of a prior obligation', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\s+am\s+busy\b', 'on account of a prior engagement', s, flags=re.I)
    s = re.sub(r'\bbecause\s+i\b', 'inasmuch as I', s, flags=re.I)
    s = re.sub(r'\bbecause\b', 'in view of the fact that', s, flags=re.I)
    s = re.sub(r'\bcannot\s+come\s+to\b', 'regret to inform you that I shall be unable to attend', s, flags=re.I)
    s = re.sub(r'\bcannot\s+come\b', 'regret to inform you that I shall be unable to attend the', s, flags=re.I)
    # Apply formal vocabulary swaps
    s = _APPLY_FORMAL(s)
    # Fix grammar issues from substitutions (including capitalizing 'i')
    s = _fix_grammar_after_substitution(s)
    # Clean up punctuation artifacts
    s = _cleanup_punctuation_artifacts(s)
    # Strip leading greeting artefacts like "Hello," before prefix
    s = re.sub(r'^(hello|hi|greetings)[,!]?\s*', '', s, flags=re.I).strip()
    # Only add prefix if sentence doesn't already start with formal phrasing
    if intensity > 60 and not re.match(r'^(dear|i\s+wish\s+to|i\s+regret|i\s+would|please|kindly|further|at\s+the\s+earliest)', s, re.I):
        s = "I wish to inform you that " + s[0].lower() + s[1:]
    # Run grammar fix again after adding prefix (to fix any 'i' that appeared)
    s = _fix_grammar_after_substitution(s)
    # Capitalize first letter and ensure proper ending
    s = _capitalise_first(s)
    s = _ensure_period(s)
    # Final cleanup pass
    s = _cleanup_punctuation_artifacts(s)
    return s


def _rewrite_sentence_casual(sent: str, intensity: int) -> str:
    s = sent.strip()
    s = re.sub(r'\bat your earliest convenience\b', 'ASAP', s, flags=re.I)
    s = re.sub(r'\bat the earliest opportunity\b', 'ASAP', s, flags=re.I)
    s = re.sub(r'\bdue to\b', 'because', s, flags=re.I)
    s = re.sub(r'\bhowever\b', 'but', s, flags=re.I)
    s = re.sub(r'\btherefore\b', 'so', s, flags=re.I)
    s = re.sub(r'\badditionally\b', 'also', s, flags=re.I)
    s = _APPLY_CASUAL(s)
    return _capitalise_first(_ensure_period(s))


def _rewrite_sentence_friendly(sent: str, intensity: int) -> str:
    s = sent.strip()
    s = re.sub(r'\bat your earliest convenience\b', 'soon', s, flags=re.I)
    s = re.sub(r'\bdue to\b', 'because', s, flags=re.I)
    s = re.sub(r'\bhowever\b', 'but', s, flags=re.I)
    s = _APPLY_FRIENDLY(s)
    return _capitalise_first(_ensure_period(s))


def _rewrite_sentence_academic(sent: str, intensity: int) -> str:
    s = sent.strip()
    # Strip greeting artefacts
    s = re.sub(r'^(Hello|Hi|Hey)[,!]?\s*', '', s, flags=re.I).strip()
    s = re.sub(r'^[,!?;\s]+', '', s).strip()
    s = _capitalise_first(s)
    s = re.sub(r'\bASAP\b', 'without undue delay', s)
    s = re.sub(r'\bas soon as possible\b', 'in an expeditious manner', s, flags=re.I)
    s = re.sub(r'\bbecause\b', 'owing to the fact that', s, flags=re.I)
    s = re.sub(r'\bdue to\b', 'as a consequence of', s, flags=re.I)
    s = re.sub(r'\bpeople think\b', 'scholars argue', s, flags=re.I)
    s = re.sub(r'\bstudies show\b', 'empirical evidence suggests', s, flags=re.I)
    s = _APPLY_ACADEMIC(s)
    if intensity > 60 and not re.match(r'^(it|this|the|empirical|research|one)', s, re.I):
        s = "It is noteworthy that " + s[0].lower() + s[1:]
    return _capitalise_first(_ensure_period(s))


def _rewrite_sentence_emotional(sent: str, intensity: int) -> str:
    s = sent.strip()
    s = re.sub(r'\.\s*$', '', s)
    # Add emotional colouring
    if intensity > 60:
        if not re.search(r'(unfortunately|sadly|heartfelt|truly|deeply|sincerely)', s, re.I):
            s = "I truly feel that " + s[0].lower() + s[1:]
    return _capitalise_first(s + "...")


def _rewrite_sentence_humorous(sent: str, intensity: int) -> str:
    s = _strip_trailing_punct(sent.strip())
    additions = [
        " — no pressure though, haha!",
        " (yes, really 😄)",
        " — because why not!",
        " 😂",
    ]
    # Rotate based on sentence content hash
    idx = sum(ord(c) for c in s) % len(additions)
    if intensity > 50:
        s = s + additions[idx]
    return _capitalise_first(s)


def _rewrite_sentence_neutral(sent: str, intensity: int) -> str:
    s = sent.strip()
    s = re.sub(r'!+', '.', s)
    s = re.sub(r'\.{2,}', '.', s)
    s = re.sub(r'(unfortunately|sadly|happily|luckily),?\s*', '', s, flags=re.I)
    s = re.sub(r'\bvery\b', '', s, flags=re.I)
    s = re.sub(r'\breally\b', '', s, flags=re.I)
    s = re.sub(r'\bextremely\b', '', s, flags=re.I)
    s = re.sub(r' {2,}', ' ', s)
    return _capitalise_first(_ensure_period(s.strip()))


_SENTENCE_REWRITERS = {
    "Professional": _rewrite_sentence_professional,
    "Formal":       _rewrite_sentence_formal,
    "Casual":       _rewrite_sentence_casual,
    "Friendly":     _rewrite_sentence_friendly,
    "Academic":     _rewrite_sentence_academic,
    "Emotional":    _rewrite_sentence_emotional,
    "Humorous":     _rewrite_sentence_humorous,
    "Neutral":      _rewrite_sentence_neutral,
}


# ---------------------------------------------------------------------------
# Step 4 — Prefix / suffix decorators per tone
# ---------------------------------------------------------------------------

def _apply_prefix_suffix(text: str, tone: str, intensity: int) -> str:
    """Add tone-appropriate greeting/closing for high-intensity transforms."""
    if intensity < 70:
        return text

    if tone == "Professional":
        # Sentence rewriter already adds "Please" for imperative sentences
        # Just ensure proper ending
        if not text.rstrip().endswith(("regards.", "sincerely.", "you.", "convenience.")):
            text = text.rstrip(".") + "."

    elif tone == "Formal":
        # Sentence rewriter already prepends "I wish to inform you that ..."
        # Only add prefix if sentence doesn't already start with formal phrasing
        if not re.match(r'^(dear|i\s+wish\s+to|i\s+regret|i\s+would|please|kindly|further|at\s+the\s+earliest|i\s+necessitate)', text, re.I):
            text = "I wish to formally bring to your attention that " + text[0].lower() + text[1:]

    elif tone == "Friendly":
        if not re.match(r'^(hi|hey|hello)', text, re.I):
            text = "Hi there! " + text
        if not re.search(r'(thanks|thank you|appreciate|hope)', text, re.I):
            text = text.rstrip(".!") + ". Hope this helps!"

    elif tone == "Humorous":
        if not text.endswith("!"):
            text = text.rstrip(".") + "!"

    elif tone == "Emotional":
        pass  # Sentence rewriter adds "I truly feel that ..."

    return text


# ---------------------------------------------------------------------------
# Core transformation pipeline
# ---------------------------------------------------------------------------

def _rule_based_transform(text: str, tone: str, intensity: int) -> str:
    """Full pipeline: normalise → sentence rewrite → decorate."""

    # Phase 1: Normalise casual language (only for formal-direction tones)
    if tone in ("Professional", "Formal", "Academic", "Neutral"):
        text = _NORMALIZE_SLANG(text)
        # Clean up empty-substitution artefacts (vocative removal may leave stray commas/spaces)
        text = re.sub(r',\s*,', ',', text)          # double comma
        text = re.sub(r'^[,!?;\s]+', '', text)       # leading punctuation
        text = re.sub(r'\s{2,}', ' ', text).strip()
        # Strip leading greeting words that survived (Hello/Hi/Hey) with trailing comma
        text = re.sub(r'^(Hello|Hi|Hey)[,!]?\s*', '', text).strip()
        text = _capitalise_first(text)

    # Phase 2: Sentence-level rewriting
    rewriter = _SENTENCE_REWRITERS.get(tone)
    if rewriter is None:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    rewritten = [rewriter(s, intensity) for s in sentences]
    result = " ".join(rewritten)

    # Phase 3: Prefix / suffix decoration
    result = _apply_prefix_suffix(result, tone, intensity)

    # Final clean-up
    result = re.sub(r'\s{2,}', ' ', result).strip()
    result = _capitalise_first(result)

    return result


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

    try:
        transformed = _rule_based_transform(text, tone, intensity)
    except Exception as exc:
        # Graceful degradation — return original text
        return {"transformed": text, "method": f"error-fallback ({exc})"}

    return {
        "transformed": transformed or text,
        "method": f"enhanced-linguistic-rules ({tone}, intensity={intensity})",
    }
