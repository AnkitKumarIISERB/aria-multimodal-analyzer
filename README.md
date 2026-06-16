# Aria - Multimodal Analyzer

Aria is an end-to-end multimodal emotion analysis pipeline that fuses acoustic embeddings with real-time facial geometry to detect internal conflict (when facial expressions mismatch vocal tone).

## Architecture Decisions

### 1. Separation of Application Logic and ML Inference (Microservices)
What you see here is a true microservice architecture. 

**Application flow:**
`Browser → FastAPI (Render) → HF Space (WavLM) → FastAPI (Render) → Browser`

The Hugging Face Space is a standalone ML inference microservice. It has no knowledge of sessions, users, or fusion logic. It simply accepts raw audio bytes via an HTTP endpoint and returns an emotion score. The core application logic (FastAPI on Render) handles the WebSocket streams, state management, and the `fusion_mode` conflict resolution. 

This separation of concerns (inference service vs application logic) is exactly how production ML systems are architected at scale (e.g., Google, Hume AI). It also solves the challenge of deploying memory-intensive models (WavLM is 1.2GB) by offloading the heavy inference to a dedicated 30GB Hugging Face instance rather than burdening the lightweight web server.

### 2. Single-Instance State Management
Session state (face and audio emotion scores) is stored in-process using a standard Python dictionary. This is correct for our single-instance deployment on Render. 

A production multi-instance deployment would replace this with a Redis pub/sub mechanism to ensure state consistency across horizontal pods. The interface here is intentionally abstracted behind the `get_state` and `set_state` functions in `main.py` to make that future Redis migration a one-line infrastructure swap without having to rewrite the core WebSocket or fusion loops.

## Deployment Setup

### 1. ML Microservice (Hugging Face Spaces)
Deploy the `/hf_space` directory to a Hugging Face Space running FastAPI.
This service exposes a `/infer` endpoint and a `/warmup` endpoint.

### 2. Application Backend (Render)
Deploy the `/backend` directory as a Render Web Service.
Required Environment Variables:
- `HF_INFERENCE_URL`: The URL of your deployed HF Space.
- `AUDIO_WEIGHT`: `0.7`
- `FACE_WEIGHT`: `0.3`
- `FACE_CONFIDENCE_THRESHOLD`: `0.6`

### 3. Frontend (Vercel)
Deploy the `/frontend` React application to Vercel.
Set the `VITE_API_URL` to point to the Render backend URL.
