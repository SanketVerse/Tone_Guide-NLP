"""ToneGuide — NLP-Based Text Tone Transformer.

FastAPI backend + plain HTML/CSS/JS frontend.
Pure NLP approach — no AI models, no API keys required.

Run locally:
    python3 app.py
    Open http://localhost:8000

Environment variables (all optional):
    PORT   Server port (default: 8000)
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure local packages always resolve correctly regardless of cwd
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nlp.preprocessing import preprocess_for_analysis
from nlp.sentiment import analyze_sentiment
from nlp.emotion import analyze_emotion
from nlp.tone import analyze_tone
from nlp.transformer import transform_text, TARGET_TONES
from nlp.context_formatter import apply_context_formatting
from nlp.similarity import compute_similarity
from nlp.marathi import (
    marathi_preprocess,
    analyze_marathi_sentiment,
    analyze_marathi_emotion,
    analyze_marathi_tone,
    transform_marathi_text,
    apply_marathi_context_formatting,
    MARATHI_TONES,
)
from utils.helpers import validate_input, truncate


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ToneGuide] Ready — Pure NLP (NLTK VADER + TF-IDF + Marathi Regional NLP). No API keys needed.")
    yield


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ToneGuide",
    description="NLP-Based Text Tone Transformer — College NLP Mini-Project with Marathi Regional Support",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class TransformRequest(BaseModel):
    text: str     = Field(..., min_length=1, description="Input text")
    tone: str     = Field("Professional", description="Target tone")
    intensity: int = Field(70, ge=0, le=100, description="Tone intensity 0–100")
    context: str  = Field("General Text", description="Output context")
    regional: bool = Field(False, description="Enable Marathi regional NLP mode")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/api/tones", summary="List available tones")
def get_tones() -> list:
    """Return the list of supported target tones."""
    return TARGET_TONES


@app.post("/api/transform", summary="Run full NLP pipeline")
def transform_endpoint(req: TransformRequest) -> dict:
    """
    Execute the complete NLP pipeline.
    - regional=true  → Pure Marathi NLP engine (lexicon + honorific rules)
    - regional=false → Standard English NLP engine (NLTK VADER + TF-IDF)
    """
    is_valid, err_msg = validate_input(req.text)
    if not is_valid:
        if not req.text.strip() or "too long" in err_msg:
            raise HTTPException(status_code=400, detail=err_msg)

    clean_text = truncate(req.text.strip())

    # Normalise tone — fall back to Neutral if caller sends an unknown value
    resolved_tone = req.tone if req.tone in TARGET_TONES else "Neutral"

    t0 = time.time()

    try:
        # ── MARATHI REGIONAL PIPELINE ─────────────────────────────────────
        if req.regional:
            orig_pre  = marathi_preprocess(clean_text)
            orig_sent = analyze_marathi_sentiment(clean_text)
            orig_emo  = analyze_marathi_emotion(clean_text)
            orig_tone = analyze_marathi_tone(clean_text, orig_pre)

            trans_res        = transform_marathi_text(clean_text, resolved_tone, req.intensity)
            transformed_text = trans_res.get("transformed", clean_text) or clean_text
            gen_method       = trans_res.get("method", "marathi-linguistic-rules")

            context_res      = apply_marathi_context_formatting(transformed_text, req.context, resolved_tone)
            transformed_text = context_res["text"]

            trans_pre  = marathi_preprocess(transformed_text)
            trans_sent = analyze_marathi_sentiment(transformed_text)
            trans_emo  = analyze_marathi_emotion(transformed_text)
            trans_tone = analyze_marathi_tone(transformed_text, trans_pre)

            sim_res    = compute_similarity(clean_text, transformed_text)
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            return {
                "original_text":      clean_text,
                "transformed_text":   transformed_text,
                "target_tone":        resolved_tone,
                "intensity":          req.intensity,
                "context":            context_res["context"],
                "formatting_applied": context_res["formatting_applied"],
                "gen_method":         gen_method,
                "regional_mode":      True,
                "language":           "Marathi (मराठी)",
                "original_analysis": {
                    "preproc":   orig_pre,
                    "sentiment": orig_sent,
                    "emotion":   orig_emo,
                    "tone":      orig_tone,
                },
                "transformed_analysis": {
                    "preproc":   trans_pre,
                    "sentiment": trans_sent,
                    "emotion":   trans_emo,
                    "tone":      trans_tone,
                },
                "similarity":  sim_res,
                "latency_ms":  elapsed_ms,
            }

        # ── ENGLISH PIPELINE ──────────────────────────────────────────────
        orig_pre  = preprocess_for_analysis(clean_text)
        orig_sent = analyze_sentiment(clean_text)
        orig_emo  = analyze_emotion(clean_text)
        orig_tone = analyze_tone(clean_text, orig_pre)

        trans_res        = transform_text(clean_text, resolved_tone, req.intensity)
        transformed_text = trans_res.get("transformed", clean_text) or clean_text
        gen_method       = trans_res.get("method", "rule-based")

        context_res      = apply_context_formatting(transformed_text, req.context, resolved_tone)
        transformed_text = context_res["text"]

        trans_pre  = preprocess_for_analysis(transformed_text)
        trans_sent = analyze_sentiment(transformed_text)
        trans_emo  = analyze_emotion(transformed_text)
        trans_tone = analyze_tone(transformed_text, trans_pre)

        sim_res    = compute_similarity(clean_text, transformed_text)
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "original_text":      clean_text,
            "transformed_text":   transformed_text,
            "target_tone":        resolved_tone,
            "intensity":          req.intensity,
            "context":            context_res["context"],
            "formatting_applied": context_res["formatting_applied"],
            "gen_method":         gen_method,
            "regional_mode":      False,
            "language":           "English",
            "original_analysis": {
                "preproc":   orig_pre,
                "sentiment": orig_sent,
                "emotion":   orig_emo,
                "tone":      orig_tone,
            },
            "transformed_analysis": {
                "preproc":   trans_pre,
                "sentiment": trans_sent,
                "emotion":   trans_emo,
                "tone":      trans_tone,
            },
            "similarity":  sim_res,
            "latency_ms":  elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


@app.get("/", include_in_schema=False)
def serve_index():
    """Serve the main HTML page."""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[ToneGuide] Starting at http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
