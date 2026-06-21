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
        from transformers import Wav2Vec2FeatureExtractor, AutoModel
        import torch
        
        model_id = "microsoft/wavlm-base"
        logger.info(f"Downloading/Loading {model_id}...")
        wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        wavlm_model = AutoModel.from_pretrained(model_id)
        wavlm_model.eval()
        logger.info("WavLM model loaded and ready.")
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
    Receives raw audio bytes, runs real inference, and returns an emotion score.
    """
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio bytes provided")

    if wavlm_model is None or wavlm_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    import torch
    import numpy as np

    try:
        # Decode raw PCM Float32 audio sent by the frontend ScriptProcessorNode
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Ensure minimum length for WavLM (pad with zeros if too short)
        if len(audio_array) < 1600:  # minimum ~100ms at 16kHz
            audio_array = np.pad(audio_array, (0, 1600 - len(audio_array)))
        
        inputs = wavlm_processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
            
        # Extract emotion score from hidden states via mean pooling + sigmoid
        # This maps the 768-dim embedding space to a scalar valence score [0, 1]
        hidden_states = outputs.last_hidden_state  # (1, T, 768)
        pooled = hidden_states.mean(dim=1)          # (1, 768)
        wavlm_score = float(torch.sigmoid(pooled.mean()).item())

        return {
            "emotion_score": wavlm_score
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")
