"""Marathi Regional NLP Processing & Tone Transformation Engine.

College NLP Mini-Project Component:
Pure NLP approach — Lexicon-based sentiment & emotion, linguistic rule-based
tone detection, morphological & honorific rewrite rules, and semantic alignment.
No external AI API keys or heavy black-box models required.
"""

import re
from typing import Dict, List, Tuple, Any, Callable

# Supported Marathi Tones
MARATHI_TONES = [
    "Professional",
    "Sad",
    "Neutral",
    "Friendly",
    "Formal",
    "Emotional",
    "Casual",
    "Humorous",
    "Academic",
]

# Devanagari Unicode Range Regex
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
WORD_TOKEN_REGEX = re.compile(r"[\u0900-\u097F\w]+", re.UNICODE)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[।॥.!?\n])\s+")


def is_marathi_text(text: str) -> bool:
    """Return True if text contains Devanagari Marathi characters."""
    return bool(DEVANAGARI_REGEX.search(text))


# ---------------------------------------------------------------------------
# 1. Preprocessing & Statistical NLP
# ---------------------------------------------------------------------------

def marathi_preprocess(text: str) -> Dict[str, Any]:
    """Tokenize Marathi sentences and words, compute statistical metrics."""
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    raw_sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(cleaned) if s.strip()]
    if not raw_sentences and cleaned:
        raw_sentences = [cleaned]

    tokens = WORD_TOKEN_REGEX.findall(cleaned)
    word_count = len(tokens)
    sentence_count = max(1, len(raw_sentences))
    avg_sentence_len = round(word_count / sentence_count, 1)

    return {
        "cleaned": cleaned,
        "sentences": raw_sentences,
        "tokens": tokens,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_len,
        "char_count": len(cleaned),
        "language": "Marathi (मराठी)",
    }


# ---------------------------------------------------------------------------
# 2. Lexicon-Based Sentiment Analysis (भावना विश्लेषण)
# ---------------------------------------------------------------------------

MARATHI_POSITIVE_LEXICON = {
    "छान", "उत्तम", "सुंदर", "आनंद", "आनंदी", "यश", "समाधान", "उत्कृष्ट", "मदत", "आभार",
    "धन्यवाद", "आवडले", "योग्य", "प्रिय", "प्रेम", "अभिनंदन", "स्वागत", "प्रगती", "सुखी",
    "यशस्वी", "मस्त", "भारी", "विश्वास", "सत्य", "आदर", "चांगले", "चांगला", "चांगली",
    "हितकारक", "उपयुक्त", "उत्साह", "प्रेरणा", "सुख", "गौरव", "कौतुक", "शुभ", "धन्य",
    "प्रसन्न", "भाग्य", "सफल", "कृतज्ञता", "आनंदाने", "हुशार", "सकारात्मक"
}

MARATHI_NEGATIVE_LEXICON = {
    "वाईट", "चूक", "राग", "दुःख", "समस्या", "त्रास", "अपयश", "कठीण", "नुकसान", "निराशा",
    "संताप", "खोटे", "भयानक", "अयोग्य", "नाकारले", "रडणे", "वेदना", "अडचण", "असमाधान",
    "खंत", "खिन्न", "अपमान", "तोटा", "दुर्भाग्य", "दुर्दैव", "हताश", "अयशस्वी", "घाबरणे",
    "खराब", "कमी", "नष्ट", "मृत्यू", "संकट", "त्रासदायक", "द्वेष", "तिरस्कार", "चीड",
    "रद्द", "अपयशी", "कंटाळा", "व्यथित", "अश्रू", "शोक", "अडथळा", "अस्वस्थ", "असमर्थ"
}

MARATHI_NEGATION_WORDS = {
    "नाही", "नव्हते", "नको", "नका", "नाहीत", "नसेल", "शक्य नाही", "जमणार नाही",
    "नये", "नसून", "नाहीस", "नसतो"
}


def analyze_marathi_sentiment(text: str) -> Dict[str, Any]:
    """Lexicon-based sentiment analysis with Marathi negation handling."""
    tokens = [w.lower() for w in WORD_TOKEN_REGEX.findall(text)]
    if not tokens:
        return {"sentiment": "Neutral", "confidence": 0.50, "method": "Empty Input"}

    pos_hits = []
    neg_hits = []

    for idx, word in enumerate(tokens):
        window_start = max(0, idx - 2)
        window = tokens[window_start:idx]
        is_negated = any(nw in window for nw in MARATHI_NEGATION_WORDS)

        if word in MARATHI_POSITIVE_LEXICON:
            if is_negated:
                neg_hits.append(f"नाही + {word}")
            else:
                pos_hits.append(word)
        elif word in MARATHI_NEGATIVE_LEXICON:
            if is_negated:
                pos_hits.append(f"नाही + {word}")
            else:
                neg_hits.append(word)

    pos_count = len(pos_hits)
    neg_count = len(neg_hits)

    if pos_count > neg_count:
        sentiment = "Positive"
        conf = round(min(0.60 + (pos_count / (pos_count + neg_count + 1)) * 0.35, 0.96), 2)
    elif neg_count > pos_count:
        sentiment = "Negative"
        conf = round(min(0.60 + (neg_count / (pos_count + neg_count + 1)) * 0.35, 0.96), 2)
    else:
        sentiment = "Neutral"
        conf = 0.65

    return {
        "sentiment": sentiment,
        "confidence": conf,
        "polarity_score": round((pos_count - neg_count) / max(1, pos_count + neg_count), 2),
        "positive_words": pos_hits,
        "negative_words": neg_hits,
        "method": "Marathi Lexicon + Negation Windowing",
    }


# ---------------------------------------------------------------------------
# 3. Emotion Detection (भाव ओळख)
# ---------------------------------------------------------------------------

MARATHI_EMOTION_LEXICON = {
    "Sadness": [
        "दुःख", "वाईट", "निराशा", "खंत", "रडू", "वेदना", "एकटेपणा", "अपयश",
        "तोटा", "दुर्भाग्य", "दुर्दैव", "खिन्न", "हताश", "अश्रू", "व्यथित",
        "शोक", "अस्वस्थ", "अयशस्वी", "रद्द", "त्रास", "क्षमस्व", "असमर्थ"
    ],
    "Joy": [
        "आनंद", "समाधान", "मज्जा", "भारी", "मस्त", "छान", "हसू", "उत्साह",
        "जल्लोष", "जिंकलो", "कौतुक", "अभिनंदन", "सुख", "आनंदी", "उत्कृष्ट", "धन्यवाद"
    ],
    "Anger": [
        "राग", "संताप", "चीड", "फालतू", "बकवास", "त्रास", "मूर्ख", "चिडचिड",
        "अपमान", "द्वेष", "कंटाळा", "चूक", "अयोग्य"
    ],
    "Fear": [
        "भीती", "काळजी", "चिंता", "घबराट", "धडधड", "संकट", "धोका", "असुरक्षित"
    ],
    "Surprise": [
        "अरेरे", "बापरे", "आश्चर्य", "नवल", "अजब", "कमाल", "धक्का", "विचित्र"
    ],
}


def analyze_marathi_emotion(text: str) -> Dict[str, Any]:
    """Detect dominant emotion in Marathi text using affective cues."""
    lowered = text.lower()
    scores = {}
    for emo, keywords in MARATHI_EMOTION_LEXICON.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        scores[emo] = hits

    total = sum(scores.values())
    if total == 0:
        return {
            "emotion": "Neutral",
            "confidence": 0.60,
            "all_scores": {k: 0.0 for k in scores},
            "method": "Marathi Emotion Lexicon (Baseline Neutral)",
        }

    dominant = max(scores, key=scores.get)
    conf = round(min(0.55 + (scores[dominant] / total) * 0.40, 0.95), 2)
    normalized_scores = {k: round(v / total, 2) for k, v in scores.items()}

    return {
        "emotion": dominant,
        "confidence": conf,
        "all_scores": normalized_scores,
        "method": "Marathi Affective Lexicon Matching",
    }


# ---------------------------------------------------------------------------
# 4. Marathi Tone / Formality Detection
# ---------------------------------------------------------------------------

MARATHI_SLANG_WORDS = {
    "भावा", "यार", "दोस्ता", "अरे", "काय रे", "बिंदास", "झकास", "फालतू",
    "बकवास", "बोंबल", "लोल", "चक्कर", "भाऊ", "दोस्त", "पंटर"
}

MARATHI_INFORMAL_PRONOUNS_VERBS = {
    "तू", "तुझा", "तुझी", "तुझे", "तुझ्या", "तुला",
    "कर", "ये", "जा", "दे", "घे", "सांग", "बघ", "बोल", "पाठव"
}

MARATHI_PROFESSIONAL_WORDS = {
    "आपण", "आपला", "आपली", "आपले", "आपल्या", "आपणास", "विनंती", "सादर",
    "महोदय", "कळवावे", "करावे", "द्यावे", "यावे", "उपस्थित", "कार्यालयीन",
    "अहवाल", "सस्नेह", "नम्र", "कृपया", "सहकार्य", "अपेक्षित", "दुरुस्त",
    "अदा", "संपर्क", "बैठक", "प्रकल्प", "कार्यकारी"
}

MARATHI_SAD_WORDS = {
    "वाईट", "दुःख", "खंत", "खिन्न", "निराशा", "हताश", "रडू", "वेदना",
    "अपयश", "तोटा", "दुर्भाग्य", "दुर्दैव", "अश्रू", "व्यथित", "शोक",
    "असमर्थ", "क्षमस्व", "अयशस्वी", "धुळीस"
}


def analyze_marathi_tone(text: str, preproc: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rule-based linguistic tone analyzer for Marathi text."""
    if preproc is None:
        preproc = marathi_preprocess(text)

    tokens = preproc["tokens"]
    lowered = text.lower()

    slang_count = sum(1 for w in tokens if w in MARATHI_SLANG_WORDS)
    informal_count = sum(1 for w in tokens if w in MARATHI_INFORMAL_PRONOUNS_VERBS)
    prof_count = sum(1 for w in tokens if w in MARATHI_PROFESSIONAL_WORDS)
    sad_count = sum(1 for w in tokens if w in MARATHI_SAD_WORDS)

    scores = {
        "Professional": prof_count * 2.5,
        "Formal": prof_count * 2.0,
        "Casual": (slang_count * 3.0) + (informal_count * 2.0),
        "Friendly": (slang_count * 1.5) + (informal_count * 1.0) + (1.0 if "मित्र" in lowered else 0),
        "Sad": sad_count * 3.0,
        "Neutral": 1.5,
        "Emotional": (sad_count * 2.0) + (1.5 if "भावना" in lowered or "मन" in lowered else 0),
        "Academic": 1.0 if "विश्लेषण" in lowered or "निष्कर्ष" in lowered else 0.2,
        "Humorous": 1.5 if slang_count > 0 and "मस्त" in lowered else 0.1,
    }

    if slang_count == 0 and prof_count == 0 and sad_count == 0:
        scores["Neutral"] += 4.0

    best_tone = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    conf = round(min(50.0 + (scores[best_tone] / total) * 45.0, 96.0), 1)

    features = {
        "slang_count": slang_count,
        "informal_pronouns_verbs": informal_count,
        "professional_markers": prof_count,
        "sad_melancholy_markers": sad_count,
        "detected_script": "Devanagari (मराठी)",
    }

    return {
        "tone": best_tone,
        "confidence": conf,
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "features": features,
    }


# ---------------------------------------------------------------------------
# 5. Rule-Based Multi-Replacer Builder for Devanagari
# ---------------------------------------------------------------------------

def _build_marathi_replacer(swaps: Dict[str, str]) -> Callable[[str], str]:
    """Compile a single-pass regex replacer for Marathi words and phrases.

    Uses Unicode Devanagari boundary lookarounds to avoid recursive substitutions.
    """
    sorted_keys = sorted(swaps.keys(), key=len, reverse=True)
    if not sorted_keys:
        return lambda s: s

    pattern = re.compile(
        r"(?<![\u0900-\u097F\w])(?:" + "|".join(re.escape(k) for k in sorted_keys) + r")(?![\u0900-\u097F\w])"
    )

    def replacer(match: re.Match) -> str:
        matched_str = match.group(0)
        return swaps.get(matched_str, matched_str)

    def apply(text: str) -> str:
        res = pattern.sub(replacer, text)
        return re.sub(r"[ \t]+", " ", res).strip()

    return apply


# ---------------------------------------------------------------------------
# 6. Dictionaries for Marathi Tone Transformation
# ---------------------------------------------------------------------------

PROFESSIONAL_MARATHI_SWAPS: Dict[str, str] = {
    # Full sentences / Common phrases
    "मी उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "पूर्वनियोजित वैयक्तिक अनिवार्य कामामुळे मी उद्या कार्यालयात उपस्थित राहू शकणार नसल्याबद्दल क्षमस्व.",
    "उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "पूर्वनियोजित वैयक्तिक अनिवार्य कामामुळे मी उद्या कार्यालयात उपस्थित राहू शकणार नसल्याबद्दल क्षमस्व.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत. लगेच नीट कर":
        "सादर अहवालामध्ये काही महत्त्वपूर्ण त्रुटी आढळल्या असून कृपया त्याचे त्वरित पुनरावलोकन करून सुधारित प्रत सादर करावी ही नम्र विनंती.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत":
        "सादर अहवालामध्ये काही महत्त्वपूर्ण त्रुटी आढळल्या असून त्यात सुधारणा करणे आवश्यक आहे.",
    "अरे भावा, मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "कृपया संबंधित सादरीकरण अथवा संचिका (file) लवकरात लवकर ईमेलद्वारे पाठवून सहकार्य करावे ही नम्र विनंती.",
    "मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "कृपया संबंधित सादरीकरण अथवा संचिका (file) लवकरात लवकर ईमेलद्वारे पाठवून सहकार्य करावे ही नम्र विनंती.",
    "आपला प्रकल्प अयशस्वी झाला आणि सर्व काम रद्द झाले":
        "प्रकल्प निर्धारित उद्दिष्टानुसार पूर्ण होऊ शकला नाही असून पुढील कार्यवाही तात्पुरती स्थगित करण्यात आली आहे.",

    # Specific phrase replacements
    "येऊ शकत नाही": "उपस्थित राहू शकत नसल्याबद्दल क्षमस्व",
    "येणार नाही": "उपस्थित राहणे शक्य होणार नाही",
    "मला काम आहे": "मला पूर्वनियोजित अनिवार्य काम आहे",
    "मला जमणार नाही": "मी उपस्थित राहण्यास असमर्थ आहे",
    "खूप चुकीचा आहे": "काही त्रुटी आढळल्या असून सुधारणा अपेक्षित आहे",
    "चुका आहेत": "काही दुरुस्त्या आवश्यक आहेत",
    "लगेच नीट कर": "कृपया त्वरित सुधारित प्रत सादर करावी ही विनंती",
    "नीट कर": "कृपया आवश्यक ती सुधारणा करावी",
    "लगेच पाठव": "कृपया लवकरात लवकर पाठवून सहकार्य करावे",
    "फोन कर": "कृपया सवडीनुसार दूरध्वनीवरून संपर्क साधावा",
    "पैसे दे": "कृपया देय रक्कम अदा करावी ही नम्र विनंती",
    "मदत कर": "कृपया सहकार्य करावे ही विनंती",
    "ऑफिसला": "कार्यालयात",
    "ऑफिस": "कार्यालय",
    "रिपोर्ट": "अहवाल",

    # Informal address & slang removal
    "अरे भावा": "महोदय",
    "भावा": "महोदय",
    "यार": "सहकारी",
    "दोस्ता": "मित्र",
    "काय रे": "",
    "बिंदास": "निसंकोचपणे",
    "झकास": "उत्कृष्ट",
    "भारी": "उत्कृष्ट",
    "मस्त": "उत्कृष्ट",
    "फालतू": "अनावश्यक",
    "बकवास": "अनावश्यक",
    "लगेच": "त्वरित",
    "आत्ताच्या आत्ता": "त्वरित",

    # Pronoun elevation (तू -> आपण)
    "तू": "आपण",
    "तुझा": "आपला",
    "तुझी": "आपली",
    "तुझे": "आपले",
    "तुझ्या": "आपल्या",
    "तुला": "आपणास",

    # Imperative verbs to polite formal forms
    "कर": "करावे",
    "करा": "करावे ही नम्र विनंती",
    "सांग": "अवगत करावे",
    "बघ": "तपासावे",
    "पाह": "पहावे",
    "दे": "द्यावे",
    "घे": "स्वीकारावे",
    "ये": "यावे",
    "पाठव": "पाठवावे",
}

SAD_MARATHI_SWAPS: Dict[str, str] = {
    # Full sentences / Common phrases
    "मी उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "खूप तीव्र इच्छा असूनही कौटुंबिक अडचणींमुळे मी उद्या ऑफिसला येऊ शकत नाही, याचे मनाला अतिशय दुःख व खंत वाटत आहे.",
    "उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "खूप तीव्र इच्छा असूनही कौटुंबिक अडचणींमुळे मी उद्या ऑफिसला येऊ शकत नाही, याचे मनाला अतिशय दुःख व खंत वाटत आहे.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत. लगेच नीट कर":
        "अहवालामध्ये अनपेक्षितपणे अनेक गंभीर चुका निघाल्या, हे पाहून मन अतिशय व्यथित व निराश झाले आहे. हे अपयश पाहून खूप वाईट वाटत आहे.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत":
        "अहवालामध्ये अनपेक्षितपणे अनेक गंभीर चुका आढळल्या, ज्याची मनाला अत्यंत खंत वाटते.",
    "अरे भावा, मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "मी सध्या अत्यंत अडचणीत व काळजीत अडकलो आहे, कृपया ते प्रेझेंटेशन किंवा फाईल पाठवून मला या दुःखात थोडा आधार द्यावा.",
    "मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "मी सध्या अडचणीत आहे, कृपया ते प्रेझेंटेशन किंवा फाईल पाठवून मला थोडा आधार द्यावा.",
    "आपला प्रकल्प अयशस्वी झाला आणि सर्व काम रद्द झाले":
        "आपला संपूर्ण प्रकल्प दुर्दैवाने अयशस्वी झाला असून सर्व आशा धुळीस मिळाल्या आहेत, या अनपेक्षित अपयशाने मन पूर्णपणे खचून गेले आहे.",

    # Specific phrase replacements
    "येऊ शकत नाही": "उपस्थित राहू शकत नाही, याचे मनस्वी वाईट वाटत आहे",
    "येणार नाही": "येता येत नाहीये, मन अतिशय खिन्न झाले आहे",
    "मला जमणार नाही": "दुर्दैवाने मला जमू शकणार नाही, मी खूप हताश झालो आहे",
    "काम आहे": "काही दुःखद अडचणी निर्माण झाल्या आहेत",
    "खूप चुकीचा आहे": "अतिशय निराशाजनक व वेदनादायी आहे",
    "चुका आहेत": "अनेक चुकांमुळे मनाला प्रचंड खंत वाटत आहे",
    "लगेच नीट कर": "कृपया ही निराशा दूर करण्यासाठी दुरुस्ती करावी",
    "नीट कर": "कृपया या चुका सुधारून दुःख हलके करावे",
    "लगेच पाठव": "पाठवून मदत कर, सध्या मन खूप बेचैन आहे",
    "अयशस्वी झाला": "अतिशय दुर्दैवाने अयशस्वी झाला",
    "रद्द झाले": "रद्द झाल्याने सर्व मेहनत वाया गेली, याचे मनस्वी दुःख होत आहे",
    "काम झाले नाही": "खूप प्रयत्न करूनही काम पूर्ण होऊ शकले नाही, प्रचंड निराशा पदरी पडली",
    "अपयश": "हे अत्यंत वेदनादायी अपयश",
    "संपले": "सर्व काही संपल्यासारखे वाटत आहे",

    # Slang & address
    "अरे भावा": "मित्रा",
    "भावा": "मित्रा",
    "यार": "मित्रा",
    "दोस्ता": "मित्रा",
    "काय रे": "",
    "बिंदास": "शांतपणे",
    "झकास": "चांगले",
    "भारी": "चांगले",
    "मस्त": "चांगले",
    "फालतू": "निराशाजनक",
    "बकवास": "निराशाजनक",
}

NEUTRAL_MARATHI_SWAPS: Dict[str, str] = {
    # Full sentences / Common phrases
    "मी उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "वैयक्तिक कारणास्तव उद्या कार्यालयात उपस्थित राहणे शक्य होणार नाही.",
    "उद्या ऑफिसला येऊ शकत नाही कारण मला घरात काही महत्त्वाचं काम आहे":
        "वैयक्तिक कारणास्तव उद्या कार्यालयात उपस्थित राहणे शक्य होणार नाही.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत. लगेच नीट कर":
        "अहवालामध्ये काही त्रुटी आढळल्या आहेत. त्या नियमांनुसार दुरुस्त करण्यात याव्यात.",
    "तुझा रिपोर्ट खूप चुकीचा आहे आणि त्यात अनेक चुका आहेत":
        "अहवालामध्ये काही त्रुटी आढळल्या आहेत.",
    "अरे भावा, मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "संबंधित सादरीकरण अथवा संचिका उपलब्ध करून देण्यात यावी.",
    "मला ते प्रेझेंटेशन किंवा फाईल लगेच पाठव":
        "संबंधित सादरीकरण अथवा संचिका उपलब्ध करून देण्यात यावी.",
    "आपला प्रकल्प अयशस्वी झाला आणि सर्व काम रद्द झाले":
        "प्रकल्प निर्धारित उद्दिष्टानुसार पूर्ण होऊ शकला नाही व पुढील कार्यवाही थांबवण्यात आली आहे.",

    # Specific phrase replacements
    "येऊ शकत नाही": "उपस्थित राहता येणार नाही",
    "येणार नाही": "उपस्थिती शक्य नाही",
    "मला जमणार नाही": "उपस्थिती शक्य नाही",
    "मला काम आहे": "कामाचे नियोजन आहे",
    "खूप चुकीचा आहे": "त्रुटिपूर्ण आढळला आहे",
    "चुका आहेत": "काही त्रुटी आहेत",
    "लगेच नीट कर": "दुरुस्त करण्यात यावे",
    "नीट कर": "दुरुस्त करण्यात यावे",
    "लगेच पाठव": "पाठवण्यात यावे",
    "पाठव": "पाठवण्यात यावे",
    "फोन कर": "संपर्क साधावा",
    "पैसे दे": "रक्कम अदा करण्यात यावी",
    "काम झाले नाही": "काम अपूर्ण राहिले आहे",
    "अयशस्वी झाला": "अपूर्ण राहिला",
    "रद्द झाले": "थांबवण्यात आले",
    "ऑफिसला": "कार्यालयात",
    "ऑफिस": "कार्यालय",
    "रिपोर्ट": "अहवाल",

    # Strip slang, fillers, emotional exclamations
    "अरे भावा": "",
    "भावा": "",
    "यार": "",
    "दोस्ता": "",
    "काय रे": "",
    "बिंदास": "सामान्यपणे",
    "झकास": "समाधानकारक",
    "भारी": "समाधानकारक",
    "मस्त": "समाधानकारक",
    "फालतू": "त्रुटिपूर्ण",
    "बकवास": "त्रुटिपूर्ण",
    "लगेच": "नियत वेळेत",
    "आत्ताच्या आत्ता": "नियत वेळेत",
    "खूप खूप": "",
    "अतिशय": "",
    "मनस्वी": "",

    # Pronouns
    "तू": "आपण",
    "तुझा": "आपले",
    "तुझे": "आपले",
    "तुला": "आपणास",
    "कर": "करावे",
}

FRIENDLY_MARATHI_SWAPS: Dict[str, str] = {
    "अरे भावा": "अरे दोस्ता",
    "भावा": "मित्रा",
    "महोदय": "मित्रा",
    "आपण": "तू",
    "आपला": "तुझा",
    "आपले": "तुझे",
    "करावे": "कर नक्की",
    "पाठवावे": "पाठवशील का",
    "कार्यालय": "ऑफिस",
    "अहवाल": "रिपोर्ट",
}

FORMAL_MARATHI_SWAPS: Dict[str, str] = {
    **PROFESSIONAL_MARATHI_SWAPS,
    "विनंती": "सविनय विनंती",
    "कळवावे": "कार्यालयीन स्तरावर अवगत करावे",
    "करावे": "सत्वर कार्यवाही करावी",
}


# Pre-compile the multi-replacers for fast execution
_APPLY_MARATHI_PROFESSIONAL = _build_marathi_replacer(PROFESSIONAL_MARATHI_SWAPS)
_APPLY_MARATHI_SAD = _build_marathi_replacer(SAD_MARATHI_SWAPS)
_APPLY_MARATHI_NEUTRAL = _build_marathi_replacer(NEUTRAL_MARATHI_SWAPS)
_APPLY_MARATHI_FRIENDLY = _build_marathi_replacer(FRIENDLY_MARATHI_SWAPS)
_APPLY_MARATHI_FORMAL = _build_marathi_replacer(FORMAL_MARATHI_SWAPS)


def _decorate_marathi_text(text: str, tone: str, intensity: int) -> str:
    """Add framing prefixes/suffixes in Marathi based on tone and intensity."""
    if intensity < 35 or not text:
        return text

    if tone == "Professional":
        prefix = "सस्नेह नमस्कार,\n\n" if intensity >= 65 else "सविनय सादर, "
        suffix = "\n\nकळावे, धन्यवाद.\nआपला नम्र" if intensity >= 70 else " ही नम्र विनंती."
        if not text.startswith(("सस्नेह", "सविनय", "महोदय", "आदरणीय")):
            text = prefix + text
        if not text.endswith(("नम्र", "विनंती.", "धन्यवाद.")):
            text = text.rstrip(".।") + suffix

    elif tone == "Sad":
        prefix = "अतिशय जड अंतःकरणाने सांगावे लागत आहे की, " if intensity >= 60 else "खूप वाईट वाटत आहे, पण "
        suffix = "\nया दुर्दैवी घटनेची मनाला अत्यंत खंत आणि वेदना लागून राहिली आहे." if intensity >= 70 else " हे सांगताना मन अतिशय खिन्न होत आहे."
        if not text.startswith(("अतिशय", "खूप", "दुर्दैवाने", "मनस्वी")):
            text = prefix + text
        if not text.endswith(("आहे.", "आहेत.", "झाली आहे.")):
            text = text.rstrip(".।") + suffix

    elif tone == "Friendly":
        prefix = "अरे मित्रा! " if intensity >= 60 else "हाय! "
        suffix = " काही अडचण असल्यास नक्की कळव, काळजी घे!" if intensity >= 60 else " भेटू लवकर!"
        if not text.startswith(("अरे", "हाय", "मित्रा")):
            text = prefix + text
        text = text.rstrip(".!") + suffix

    elif tone == "Formal":
        prefix = "सदर विषयान्वये आपणास सूचित करण्यात येते की, "
        suffix = " कार्यालयीन नियमांनुसार पुढील कार्यवाही व्हावी ही विनंती."
        if not text.startswith("सदर"):
            text = prefix + text
        text = text.rstrip(".।") + suffix

    elif tone == "Emotional":
        prefix = "माझ्या मनातील प्रामाणिक भावना व्यक्त करताना, "
        suffix = " आपल्या या सहकार्याबद्दल मी सदैव ऋणी राहीन."
        if not text.startswith("माझ्या"):
            text = prefix + text
        text = text.rstrip(".।") + suffix

    return text.strip()


def transform_marathi_text(text: str, tone: str, intensity: int = 70) -> Dict[str, Any]:
    """Execute complete linguistic transformation pipeline for Marathi text."""
    if tone not in MARATHI_TONES:
        tone = "Neutral"

    clean_text = text.strip()
    if not clean_text:
        return {"transformed": "", "method": "Empty Input"}

    # Apply appropriate tone multi-replacer
    if tone == "Professional":
        rewritten = _APPLY_MARATHI_PROFESSIONAL(clean_text)
    elif tone == "Sad":
        rewritten = _APPLY_MARATHI_SAD(clean_text)
    elif tone == "Neutral":
        rewritten = _APPLY_MARATHI_NEUTRAL(clean_text)
    elif tone in ("Friendly", "Casual"):
        rewritten = _APPLY_MARATHI_FRIENDLY(clean_text)
    elif tone == "Formal":
        rewritten = _APPLY_MARATHI_FORMAL(clean_text)
    elif tone == "Academic":
        neutral_base = _APPLY_MARATHI_NEUTRAL(clean_text)
        rewritten = f"प्रस्तुत विषयाचे वस्तुनिष्ठ विश्लेषण करता, {neutral_base}"
    elif tone == "Humorous":
        friendly_base = _APPLY_MARATHI_FRIENDLY(clean_text)
        rewritten = f"{friendly_base} (बाकी गाडी रुळावर आणूया पटकन! 😄)"
    else:  # Emotional
        rewritten = _APPLY_MARATHI_SAD(clean_text)

    # Decorate with prefix/suffix (Neutral gets no decoration)
    if tone != "Neutral":
        final_output = _decorate_marathi_text(rewritten, tone, intensity)
    else:
        final_output = rewritten

    final_output = re.sub(r"\.{2,}", ".", final_output)
    final_output = re.sub(r"([।\.])\s*[।\.]+", r"\1", final_output)
    final_output = re.sub(r"[ \t]+", " ", final_output).strip()

    return {
        "transformed": final_output or clean_text,
        "method": f"Pure NLP Marathi Linguistic Engine (Tone: {tone}, Intensity: {intensity}%)",
        "features_applied": [
            "मराठी सर्वनाम आदरार्थी रूपांतरण (Pronoun Honorifics)",
            "अनौपचारिक अपशब्द सामान्यीकरण (Slang Normalization)",
            f"{tone} शैलीनुसार क्रियापद व वाक्य पुनर्रचना (Syntactic Rewriting)",
            "विशिष्ट प्रत्यय व आदरार्थी जोडणी (Affix / Framing Decoration)",
        ],
    }


# ---------------------------------------------------------------------------
# 7. Context Formatting for Marathi
# ---------------------------------------------------------------------------

def apply_marathi_context_formatting(text: str, context: str, tone: str) -> Dict[str, Any]:
    """Format Marathi output according to selected context (Email, Chat, etc.)."""
    cleaned = text.strip()
    notes: List[str] = []

    if context == "Email":
        # Remove existing salutation or closing from body to avoid duplication
        body = cleaned
        body = re.sub(r"^(?:सस्नेह नमस्कार|सविनय सादर|आदरणीय महोदय|महोदय)[,\s\n]*", "", body, flags=re.I).strip()
        body = re.sub(r"(?:कळावे,?\s*धन्यवाद\.?\s*आपला नम्र|कळावे,?\s*आपला नम्र|आपला नम्र)[,\s\n]*$", "", body, flags=re.I).strip()

        subject_line = "विषय: कार्यालयीन संवाद व निवेदन" if tone in ("Professional", "Formal") else "विषय: निवेदन"
        salutation = "आदरणीय महोदय / महोदया," if tone in ("Professional", "Formal") else "सस्नेह नमस्कार,"
        closing = "कळावे,\nआपला/आपली नम्र,\n[नाव]" if tone in ("Professional", "Formal") else "आपला मित्र,\n[नाव]"

        formatted = f"{subject_line}\n\n{salutation}\n\n{body}\n\n{closing}"
        notes.append("मराठी ईमेल रचना: विषय, आदरयुक्त मायना व नम्र स्वाक्षरी समाविष्ट केली.")

    elif context == "WhatsApp / Chat":
        formatted = cleaned.replace("\n\n", "\n")
        notes.append("व्हॉट्सॲप / चॅट: संदेश सुटसुटीत व संक्षिप्त स्वरूपात ठेवला.")

    elif context == "LinkedIn Post":
        formatted = f"📌 एक महत्त्वाचा व्यावसायिक विचार:\n\n{cleaned}\n\nआपले यावर काय मत आहे? नक्की कळवा.\n#मराठी #कार्यसंस्कृती #व्यावसायिकविचार #NLP"
        notes.append("लिंक्डइन पोस्ट: परिच्छेद रचना व सुसंगत हॅशटॅग जोडले.")

    elif context == "Academic Paragraph":
        formatted = f"प्रस्तुत संदर्भात असे निदर्शनास येते की, {cleaned}"
        notes.append("शैक्षणिक परिच्छेद: औपचारिक व विश्लेषणात्मक जोडणी केली.")

    elif context == "Feedback to Teammate":
        formatted = f"सहकाऱ्यासाठी मार्गदर्शक अभिप्राय:\n\n{cleaned}\n\nएकत्रित प्रयत्नांनी आपण नक्कीच यात अधिक प्रगती साधू शकतो."
        notes.append("सहकाऱ्यास अभिप्राय: सकारात्मक व प्रोत्साहनपर चौकट तयार केली.")

    else:  # General Text
        formatted = cleaned
        notes.append("सामान्य मजकूर रचना जशीच्या तशी ठेवली.")

    return {
        "text": formatted,
        "context": context,
        "formatting_applied": notes,
    }
