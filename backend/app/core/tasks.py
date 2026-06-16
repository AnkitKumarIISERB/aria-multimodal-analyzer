from app.core.celery_app import celery_app
import time
import json
import os
import redis
from celery.signals import worker_process_init

@worker_process_init.connect
def init_worker(**kwargs):
    """
    Downloads and caches the 1.2GB WavLM model weights dynamically at worker startup.
    This prevents the model from bloating the Docker image and utilizes Render's persistent disk.
    """
    try:
        from huggingface_hub import snapshot_download
        model_id = os.environ.get("HF_MODEL_ID", "microsoft/wavlm-base")
        cache_dir = os.environ.get("HF_CACHE_DIR", "/app/model_cache")
        print(f"[Celery Init] Downloading/verifying model {model_id} to {cache_dir}...")
        snapshot_download(repo_id=model_id, cache_dir=cache_dir)
        print("[Celery Init] Model ready.")
    except ImportError:
        print("[Celery Init] huggingface_hub not installed. Skipping model download for mock execution.")
    except Exception as e:
        print(f"[Celery Init] Error loading model: {e}")

# Define the synchronous Redis client for the Celery worker
redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

@celery_app.task(name="process_audio_chunk")
def process_audio_chunk(session_id: str, audio_bytes: bytes):
    """
    Receives raw PCM audio chunks and runs the PyTorch WavLM inference.
    Because this is CPU bound, it runs safely inside the Celery worker.
    """
    try:
        # 1. Initialize PyTorch / librosa here (mocked for skeleton)
        # In the real deployment, we will load the fine-tuned RAVDESS WavLM weights
        
        # Simulate PyTorch inference time (e.g., 200ms)
        time.sleep(0.2)
        
        # Dummy WavLM extraction score
        wavlm_score = 0.65 

        # 2. Write the result directly to Redis so the FastAPI fusion loop can read it
        audio_state = {
            "score": wavlm_score,
            "timestamp": time.time()
        }
        
        redis_client.set(f"session:{session_id}:audio", json.dumps(audio_state))
        print(f"[Celery] Successfully processed chunk for {session_id} -> Score: {wavlm_score}")
        
        return "success"
    except Exception as e:
        print(f"[Celery] Audio processing failed: {e}")
        return str(e)
