"""ToneGuide — NLP-Based Text Tone Transformer.

FastAPI backend + plain HTML/CSS/JS frontend.

Run locally:
    python app.py
    Open http://localhost:8000

Environment variables (all optional):
    PORT            Server port (default: 8000)
    TONE_API_URL    OpenAI-compatible API URL (enables API generation)
    TONE_API_KEY    API key for the above
    TONE_API_MODEL  Model name (default: gpt-4o-mini)
"""

import sys
import time
import threading
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure local packages (models/, nlp/, utils/) always resolve correctly
# regardless of the working directory the user runs the script from.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nlp.preprocessing import preprocess_for_analysis
from nlp.sentiment import analyze_sentiment
from nlp.emotion import analyze_emotion
from nlp.tone import analyze_tone
from nlp.transformer import transform_text, TARGET_TONES
from nlp.similarity import compute_similarity
from utils.helpers import validate_input, truncate


# ---------------------------------------------------------------------------
# Model warm-up (runs once in a background thread at startup)
# ---------------------------------------------------------------------------

def _warmup_models() -> None:
    """Pre-warm all NLP models so the very first user request is fast."""
    dummy = "Hello, this is a warm-up sentence."
    print("[ToneGuide] Warming up NLP models in background...")
    t0 = time.time()
    try:
        analyze_sentiment(dummy)
        analyze_emotion(dummy)
        compute_similarity(dummy, dummy)
        transform_text(dummy, "Professional", 50)
        print(f"[ToneGuide] Models ready in {time.time() - t0:.1f}s")
    except Exception as exc:  # warmup failures must never crash the server
        print(f"[ToneGuide] Warmup warning (non-fatal): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup_models, daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ToneGuide",
    description="NLP-Based Text Tone Transformer — College Mini-Project",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TransformRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to transform")
    tone: str = Field("Professional", description="Target tone")
    intensity: int = Field(70, ge=0, le=100, description="Tone intensity 0–100")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/tones", summary="List available tones")
def get_tones() -> list:
    """Return the list of supported target tones."""
    return TARGET_TONES


@app.post("/api/transform", summary="Run full NLP pipeline")
def transform_endpoint(req: TransformRequest) -> dict:
    """
    Execute the complete NLP pipeline:
    1. Text preprocessing & statistics
    2. Sentiment / Emotion / Tone analysis of original text
    3. Tone transformation
    4. Re-analysis of transformed text
    5. Semantic similarity scoring
    """
    is_valid, err_msg = validate_input(req.text)
    if not is_valid:
        # Empty or too-long inputs are hard errors; short inputs proceed with warning
        if not req.text.strip() or "too long" in err_msg:
            raise HTTPException(status_code=400, detail=err_msg)

    clean_text = truncate(req.text.strip())
    t0 = time.time()

    try:
        # Step 1: Preprocessing & original NLP analysis
        orig_pre = preprocess_for_analysis(clean_text)
        orig_sent = analyze_sentiment(clean_text)
        orig_emo = analyze_emotion(clean_text)
        orig_tone = analyze_tone(clean_text, orig_pre)

        # Step 2: Tone transformation
        trans_res = transform_text(clean_text, req.tone, req.intensity)
        transformed_text = trans_res.get("transformed", clean_text)
        gen_method = trans_res.get("method", "rule-based")

        # Guard against empty transformation output
        if not transformed_text.strip():
            transformed_text = clean_text
            gen_method = "passthrough (model returned empty)"

        # Step 3: Re-analysis of transformed text
        trans_pre = preprocess_for_analysis(transformed_text)
        trans_sent = analyze_sentiment(transformed_text)
        trans_emo = analyze_emotion(transformed_text)
        trans_tone = analyze_tone(transformed_text, trans_pre)

        # Step 4: Semantic similarity
        sim_res = compute_similarity(clean_text, transformed_text)

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "original_text": clean_text,
            "transformed_text": transformed_text,
            "target_tone": req.tone,
            "intensity": req.intensity,
            "gen_method": gen_method,
            "original_analysis": {
                "preproc": orig_pre,
                "sentiment": orig_sent,
                "emotion": orig_emo,
                "tone": orig_tone,
            },
            "transformed_analysis": {
                "preproc": trans_pre,
                "sentiment": trans_sent,
                "emotion": trans_emo,
                "tone": trans_tone,
            },
            "similarity": sim_res,
            "latency_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the main HTML page."""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"[ToneGuide] Starting server at http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
