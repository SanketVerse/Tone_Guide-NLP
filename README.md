# ToneGuide — NLP-Based Text Tone Transformer

A college NLP mini-project that accepts text, analyses its linguistic
characteristics, rewrites it in a chosen target tone, and compares the
original vs. transformed result.

**No AI model. No API key. 100% classical NLP.**

---

## What It Does

| Step | NLP Concept | Method |
|------|-------------|--------|
| 1 | Text Preprocessing & Tokenization | NLTK (punkt tokenizer) |
| 2 | Sentiment Analysis | NLTK VADER (lexicon + rules) |
| 3 | Emotion Detection | Keyword-lexicon scoring |
| 4 | Tone Classification | Linguistic feature scoring (rule-based) |
| 5 | Text Transformation | Word-substitution + sentence-level rules |
| 6 | Similarity Scoring | TF-IDF + Cosine Similarity (scikit-learn) |

---

## Project Structure

```
ToneGuide/
├── app.py               # FastAPI server + API endpoints
├── index.html           # Frontend (plain HTML / CSS / JS)
├── requirements.txt     # Python dependencies
├── README.md
├── nlp/
│   ├── __init__.py
│   ├── preprocessing.py # Text normalisation, sentence split, tokenisation
│   ├── sentiment.py     # Sentiment: Positive / Negative / Neutral (VADER)
│   ├── emotion.py       # Emotion: 7-category keyword detection
│   ├── tone.py          # Tone: rule-based linguistic feature scoring (8 tones)
│   ├── transformer.py   # Tone rewriting: word substitution + sentence rules
│   └── similarity.py    # Similarity: TF-IDF cosine (scikit-learn)
└── utils/
    ├── __init__.py
    └── helpers.py       # Input validation, truncation, formatting
```

---

## Installation & Running

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Start the server
python3 app.py

# 3. Open in browser
http://localhost:8000
```

> NLTK data (`punkt`, `vader_lexicon`) downloads automatically on first run.

Interactive API docs: **http://localhost:8000/docs**

---

## NLP Concepts Demonstrated

- **Tokenization** — splitting text into sentences and words using NLTK
- **Text Normalization** — unicode cleanup, whitespace normalisation
- **Sentiment Analysis** — NLTK VADER: compound score → Positive / Negative / Neutral
- **Emotion Detection** — keyword-lexicon hit counting across 7 emotion categories
- **Linguistic Feature Extraction** — contractions, slang, formal/academic vocabulary, pronouns, punctuation, sentence length
- **Rule-Based Tone Classification** — per-tone scoring from interpretable features; fully explainable
- **Lexical Text Transformation** — word-substitution dictionaries + tone-specific sentence patterns
- **TF-IDF Vectorisation** — converting text to weighted term-frequency vectors
- **Cosine Similarity** — `cos(a, b) = (a · b) / (‖a‖ · ‖b‖)` to measure meaning preservation

---

## Limitations

- Tone detection is heuristic — may misclassify ambiguous text
- Transformation is lexical only — sentence grammar is not restructured
- Sarcasm and irony are not handled
- TF-IDF similarity is bag-of-words, not deep semantic similarity
- Works best with English text
