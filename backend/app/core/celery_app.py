from celery import Celery
import os
from celery.signals import worker_process_init

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "aria_worker",
    broker=redis_url,
    backend=redis_url
)

@worker_process_init.connect
def init_worker(**kwargs):
    """
    Pre-download DeepFace weights on worker startup to prevent Render cold start timeouts.
    """
    print("[Celery Worker] Pre-downloading DeepFace weights...")
    try:
        from deepface import DeepFace
        # Run a dummy extraction to force model download
        import numpy as np
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
        print("[Celery Worker] DeepFace weights ready.")
    except Exception as e:
        print(f"[Celery Worker] DeepFace init error: {e}")

@celery_app.task
def dummy_task():
    return "Dummy"
