import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aria HF Space Audio Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model state
wavlm_model = None
wavlm_processor = None

@app.on_event("startup")
async def load_model():
    """
    Loads the WavLM model once at Space startup.
    This prevents per-request cold starts which would cause 30s+ latency.
    """
    global wavlm_model, wavlm_processor
    logger.info("Loading WavLM model and processor...")
    
    try:
        from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
        # Example loading real weights:
        # model_id = "microsoft/wavlm-base"
        # wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        # wavlm_model = AutoModelForAudioClassification.from_pretrained(model_id)
        logger.info("WavLM model loaded and ready.")
    except ImportError:
        logger.warning("Transformers library not found. Falling back to mock ML inference for local testing.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": wavlm_model is not None
    }

@app.get("/warmup")
def warmup():
    """Endpoint used by main backend to keep this space awake."""
    return {"status": "warm"}

@app.post("/infer")
async def infer_audio(request: Request):
    """
    Receives raw audio bytes, runs inference, and returns an emotion score.
    """
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio bytes provided")

    # In a real scenario, convert raw bytes to waveform using librosa/soundfile,
    # then pass to wavlm_processor and wavlm_model.
    
    # 1. Simulate inference delay
    time.sleep(0.2)
    
    # 2. Extract score (mocked for now, just like previous Celery worker)
    # The true WavLM fine-tuned model would yield the probability of positive valence
    wavlm_score = 0.65 

    return {
        "emotion_score": wavlm_score
    }
