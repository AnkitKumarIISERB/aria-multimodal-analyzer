import asyncio
import json
import time
import os
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Aria API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Config
USE_REAL_CELERY = os.getenv("USE_CELERY", "false").lower() == "true"
AUDIO_WEIGHT = float(os.getenv("AUDIO_WEIGHT", "0.7"))
FACE_WEIGHT = float(os.getenv("FACE_WEIGHT", "0.3"))
FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.6"))

# State Management Abstraction
if USE_REAL_CELERY:
    import redis.asyncio as redis
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    redis_client = redis.from_url(redis_url, decode_responses=True)
else:
    active_sessions = {}

async def set_state(session_id: str, key: str, value_dict: dict):
    if USE_REAL_CELERY:
        await redis_client.set(f"session:{session_id}:{key}", json.dumps(value_dict))
    else:
        if session_id not in active_sessions:
            active_sessions[session_id] = {}
        active_sessions[session_id][key] = value_dict

async def get_state(session_id: str, key: str) -> dict:
    if USE_REAL_CELERY:
        data_str = await redis_client.get(f"session:{session_id}:{key}")
        return json.loads(data_str) if data_str else {}
    else:
        return active_sessions.get(session_id, {}).get(key, {})

async def clear_state(session_id: str, key: str):
    if USE_REAL_CELERY:
        await redis_client.delete(f"session:{session_id}:{key}")
    else:
        if session_id in active_sessions and key in active_sessions[session_id]:
            del active_sessions[session_id][key]

# Load conflict model
conflict_model = None
try:
    conflict_model = joblib.load('app/core/conflict_model.pkl')
    print("Conflict model loaded successfully.")
except Exception as e:
    print(f"Failed to load conflict model: {e}")

FUSION_MODE_FULL = "full"
FUSION_MODE_AUDIO_ONLY = "audio_only"
FUSION_MODE_FACE_ONLY = "face_only"
FUSION_MODE_DEGRADED = "degraded"

async def mock_audio_worker(session_id: str, data: bytes):
    """Mocks the Celery worker running WavLM"""
    await asyncio.sleep(0.2) # Simulate PyTorch inference time
    wavlm_score = 0.65 # Dummy WavLM score
    
    await set_state(session_id, "audio", {
        "score": wavlm_score,
        "timestamp": time.time()
    })
    # print(f"[Mock Celery] Processed audio chunk for {session_id} -> Score: {wavlm_score}")


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.websocket("/ws/stream/audio/{session_id}")
async def audio_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[Audio WS] Session {session_id} connected.")
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if USE_REAL_CELERY:
                from app.core.tasks import process_audio_chunk
                process_audio_chunk.delay(session_id, data)
            else:
                asyncio.create_task(mock_audio_worker(session_id, data))
            
    except WebSocketDisconnect:
        print(f"[Audio WS] Session {session_id} disconnected.")
        await clear_state(session_id, "audio")

@app.websocket("/ws/stream/video/{session_id}")
async def video_stream(websocket: WebSocket, session_id: str):
    """
    Video/Landmark WS that also handles the 500ms fusion loop.
    """
    await websocket.accept()
    print(f"[Video WS] Session {session_id} connected.")
    
    # Separate async task for the decoupled fusion loop
    async def fusion_loop():
        try:
            while True:
                await asyncio.sleep(0.5) # 500ms interval
                
                face_data = await get_state(session_id, "face")
                audio_data = await get_state(session_id, "audio")
                
                current_time = time.time()
                
                # Check for staleness (> 2 seconds old)
                face_stale = (current_time - face_data.get("timestamp", 0)) > 2.0
                audio_stale = (current_time - audio_data.get("timestamp", 0)) > 2.0
                
                # Check confidence
                face_conf = face_data.get("confidence", 1.0)
                face_unreliable = face_stale or (face_conf < FACE_CONFIDENCE_THRESHOLD)
                
                # Determine Fusion Mode
                fusion_mode = FUSION_MODE_DEGRADED
                if not face_unreliable and not audio_stale:
                    fusion_mode = FUSION_MODE_FULL
                elif not face_unreliable:
                    fusion_mode = FUSION_MODE_FACE_ONLY
                elif not audio_stale:
                    fusion_mode = FUSION_MODE_AUDIO_ONLY
                
                # Fusion logic
                fused_score = 0.0
                conflict_score = 0.0
                
                if fusion_mode == FUSION_MODE_FULL:
                    a_score = audio_data.get("score", 0)
                    f_score = face_data.get("score", 0)
                    fused_score = (a_score * AUDIO_WEIGHT) + (f_score * FACE_WEIGHT)
                    
                    # Calculate conflict using ML model
                    if conflict_model:
                        try:
                            # Model takes [audio_score, face_score, abs_diff]
                            abs_diff = abs(a_score - f_score)
                            X = np.array([[a_score, f_score, abs_diff]])
                            conflict_prob = conflict_model.predict_proba(X)[0][1] # Probability of conflict (class 1)
                            conflict_score = float(conflict_prob)
                        except Exception as e:
                            print(f"Error predicting conflict: {e}")
                            
                elif fusion_mode == FUSION_MODE_FACE_ONLY:
                    fused_score = face_data.get("score", 0)
                elif fusion_mode == FUSION_MODE_AUDIO_ONLY:
                    fused_score = audio_data.get("score", 0)
                    
                # Send fused result back to frontend
                await websocket.send_json({
                    "type": "fusion_result",
                    "fused_score": fused_score,
                    "face_score": face_data.get("score", 0) if not face_unreliable else None,
                    "audio_score": audio_data.get("score", 0) if not audio_stale else None,
                    "conflict_score": conflict_score if fusion_mode == FUSION_MODE_FULL else None,
                    "fusion_mode": fusion_mode,
                    "face_stale": face_stale,
                    "audio_stale": audio_stale
                })
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Fusion loop error: {e}")

    loop_task = asyncio.create_task(fusion_loop())
    
    try:
        while True:
            payload_str = await websocket.receive_text()
            payload = json.loads(payload_str)
            
            # Check if it's a pause signal from Visibility API
            if payload.get("type") == "video_paused":
                # User hid the tab
                await set_state(session_id, "face", {
                    "timestamp": 0 # Force stale
                })
                continue
                
            # Regular landmark payload
            landmarks = payload.get("landmarks", [])
            face_confidence = payload.get("face_confidence", 1.0)
            
            # Calculate real face emotion score from geometry
            face_score = 0.0
            if landmarks:
                from app.core.geometry import extract_face_emotion
                face_score = extract_face_emotion(landmarks)

            await set_state(session_id, "face", {
                "score": face_score,
                "confidence": face_confidence,
                "timestamp": time.time()
            })
            
    except WebSocketDisconnect:
        print(f"[Video WS] Session {session_id} disconnected.")
        loop_task.cancel()
        await clear_state(session_id, "face")
