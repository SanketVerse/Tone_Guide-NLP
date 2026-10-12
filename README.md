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
| 6 | Output Context Formatting | Rule-based structural framing (Email, Chat, Post, etc.) |
| 7 | Similarity Scoring | Blended: Content Coverage + Retention + TF-IDF + Char N-gram |

---

## Project Structure

```
ToneGuide/
├── app.py                  # FastAPI server + API endpoints
├── index.html              # Frontend (plain HTML / CSS / JS)
├── requirements.txt        # Python dependencies
├── README.md
├── docs/                   # Project guide and viva explanation documents
│   ├── ToneGuide_Project_Guide.pdf
│   └── ToneGuide_Project_Explanation_and_Viva_Guide.docx
├── nlp/
│   ├── __init__.py
│   ├── preprocessing.py    # Text normalisation, sentence split, tokenisation
│   ├── sentiment.py        # Sentiment: Positive / Negative / Neutral (VADER)
│   ├── emotion.py          # Emotion: 7-category keyword detection
│   ├── tone.py             # Tone: rule-based linguistic feature scoring (8 tones)
│   ├── transformer.py      # Tone rewriting: word substitution + sentence rules
│   ├── context_formatter.py# Context framing: Email, Chat, LinkedIn, Academic, Feedback
│   └── similarity.py       # Similarity: Blended TF-IDF + n-gram + synonym matching
└── utils/
    ├── __init__.py
    └── helpers.py          # Input validation, truncation, formatting
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
- **Context-Specific Formatting** — rule-based structural framing adapting text for Emails, WhatsApp/Chat, LinkedIn, Academic paragraphs, and Teammate feedback
- **TF-IDF Vectorisation & Blended Similarity** — combines TF-IDF cosine, key content word coverage, synonym recognition, and n-gram overlap to evaluate meaning preservation

---

## Limitations

- Tone detection is heuristic — may misclassify ambiguous text
- Transformation is lexical only — sentence grammar is not restructured
- Sarcasm and irony are not handled
- TF-IDF similarity is bag-of-words, not deep semantic similarity
- Works best with English text
