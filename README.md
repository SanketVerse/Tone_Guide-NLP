# ToneGuide — NLP-Based Text Tone Transformer

A college NLP mini-project that accepts user text, analyzes its linguistic characteristics using Natural Language Processing, rewrites it in a selected target tone, and compares the original vs. transformed text.

---

## What It Does

| Step | NLP Concept | Technology |
|------|------------|------------|
| 1 | Text Preprocessing & Tokenization | NLTK |
| 2 | Sentiment Analysis | DistilBERT (SST-2) |
| 3 | Emotion Detection | DistilRoBERTa |
| 4 | Tone Classification | Linguistic Feature Scoring (Rule-Based) |
| 5 | Text Transformation | FLAN-T5 / Linguistic Rules |
| 6 | Semantic Similarity | Sentence Transformers + Cosine Similarity |

---

## Project Structure

```
ToneGuide/
├── app.py               # FastAPI web server
├── index.html           # Frontend (plain HTML / CSS / JS)
├── requirements.txt     # Python dependencies
├── README.md
├── models/
│   ├── __init__.py
│   └── model_loader.py  # Cached HuggingFace model loaders
├── nlp/
│   ├── __init__.py
│   ├── preprocessing.py # Text normalization, sentence split, tokenization (NLTK)
│   ├── sentiment.py     # Sentiment: Positive / Negative / Neutral (DistilBERT SST-2)
│   ├── emotion.py       # Emotion: 7-class classification (DistilRoBERTa)
│   ├── tone.py          # Tone: rule-based linguistic feature scoring
│   ├── transformer.py   # Tone rewriting: FLAN-T5 → linguistic-rule fallback
│   └── similarity.py    # Semantic similarity: MiniLM embeddings + cosine
└── utils/
    ├── __init__.py
    └── helpers.py       # Input validation, truncation, formatting
```

---

## Models Used

| Task | Model | Notes |
|------|-------|-------|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | 2-class → neutral mapped via confidence margin |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 7 real labels: anger, disgust, fear, joy, neutral, sadness, surprise |
| Generation | `google/flan-t5-small` | Instruction-tuned text-to-text, CPU-friendly |
| Similarity | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim dense embeddings + cosine similarity |
| Tone | Rule-based (`nlp/tone.py`) | Fully explainable — uses linguistic features |

---

## Installation & Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/ToneGuide.git
cd ToneGuide
```

**2. Create a virtual environment**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://localhost:8000
```

> NLTK data (`punkt`, `vader_lexicon`) downloads automatically on first run — no manual step needed.

---

## Optional: Use an External Generation API

If your machine cannot run FLAN-T5 locally, you can point ToneGuide at any OpenAI-compatible API:

```bash
export TONE_API_URL="https://api.openai.com/v1/chat/completions"
export TONE_API_KEY="sk-..."
export TONE_API_MODEL="gpt-4o-mini"   # optional, defaults to gpt-4o-mini
```

NLP analysis (sentiment / emotion / tone / similarity) always runs locally.

---

## NLP Concepts Demonstrated

- **Tokenization** — splitting text into sentences and words using NLTK.
- **Text Normalization** — unicode cleanup, whitespace normalization.
- **Sentiment Analysis** — Transformer sequence classification (DistilBERT + SST-2).
- **Emotion Classification** — Transformer multi-class classification (DistilRoBERTa, 7 emotions).
- **Linguistic Feature Extraction** — counting contractions, slang, formal/academic/professional vocabulary, pronouns, punctuation marks, emojis, sentence length.
- **Rule-Based Tone Classification** — per-tone scoring from interpretable features; fully explainable.
- **Text Generation** — instruction-prompted FLAN-T5 (`text2text-generation`) for tone rewriting.
- **Sentence Embeddings** — dense 384-dim vectors from `all-MiniLM-L6-v2`.
- **Cosine Similarity** — `cos(a, b) = (a · b) / (‖a‖ · ‖b‖)` on embeddings to measure meaning preservation.

---

## Limitations

- FLAN-T5-small sometimes returns near-identical output; the app falls back to transparent linguistic rules.
- The sentiment model natively outputs 2 classes; Neutral is mapped via a documented confidence margin.
- The emotion model has **no "Love" label** — the 7 real labels are used as-is.
- First model load takes 10–20 seconds (downloads ~400 MB on first run only).

---

## Viva Quick Reference

| Question | Answer location |
|----------|----------------|
| How is sentiment detected? | `nlp/sentiment.py` — DistilBERT SST-2 + neutral confidence margin |
| How is emotion detected? | `nlp/emotion.py` — DistilRoBERTa 7-class classifier |
| How is tone classified? | `nlp/tone.py` — linguistic feature scoring (contractions, slang, formal/academic/professional vocab) |
| How is text transformed? | `nlp/transformer.py` — FLAN-T5 instruction prompt → rule-based fallback |
| How is similarity measured? | `nlp/similarity.py` — MiniLM sentence embeddings + cosine similarity |
| How are models cached? | `models/model_loader.py` — `@lru_cache` with thread-safe lock |
