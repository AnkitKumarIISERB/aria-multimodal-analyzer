import asyncio
import websockets
import json
import time

async def test_degradation():
    session_id = "test-session-999"
    audio_uri = f"ws://localhost:8000/ws/stream/audio/{session_id}"
    video_uri = f"ws://localhost:8000/ws/stream/video/{session_id}"
    
    print("Testing 4-state Graceful Degradation Chain...")
    
    try:
        async with websockets.connect(video_uri) as video_ws, \
                   websockets.connect(audio_uri) as audio_ws:
            
            # State 1: Full Fusion (Both active and confident)
            print("\n[Test 1] Emitting healthy Face and Audio...")
            for _ in range(3):
                await audio_ws.send(b"dummy_pcm")
                await video_ws.send(json.dumps({
                    "type": "landmarks",
                    "landmarks": [{"x":0, "y":0, "z":0}], # dummy
                    "face_confidence": 0.95
                }))
                await asyncio.sleep(0.5)
            
            resp = json.loads(await video_ws.recv())
            assert resp["fusion_mode"] == "full", f"Expected 'full', got {resp['fusion_mode']}"
            print("✅ State 1 Passed: fusion_mode is 'full'")
            
            # State 2: Face Only Degraded (Confidence drops below 0.6)
            print("\n[Test 2] Dropping face confidence to 0.4 (simulating hand over camera)...")
            for _ in range(3):
                await audio_ws.send(b"dummy_pcm")
                await video_ws.send(json.dumps({
                    "type": "landmarks",
                    "landmarks": [{"x":0, "y":0, "z":0}],
                    "face_confidence": 0.40 # low confidence
                }))
                await asyncio.sleep(0.5)
            
            # Flush queue to get latest
            while True:
                resp = json.loads(await video_ws.recv())
                if resp["fusion_mode"] == "audio_only":
                    break
            print("✅ State 2 Passed: fusion_mode switched to 'audio_only'")
            assert resp["face_score"] is None, "Face score should be null"
            assert resp["conflict_score"] is None, "Conflict score should be null"
            
            # State 3: Both Degraded (Audio goes silent > 2s)
            print("\n[Test 3] Stopping audio transmission for 2.5 seconds...")
            for _ in range(5):
                await video_ws.send(json.dumps({
                    "type": "landmarks",
                    "landmarks": [{"x":0, "y":0, "z":0}],
                    "face_confidence": 0.40
                }))
                await asyncio.sleep(0.5)
                
            while True:
                resp = json.loads(await video_ws.recv())
                if resp["fusion_mode"] == "degraded":
                    break
            print("✅ State 3 Passed: fusion_mode switched to 'degraded'")
            
            # State 3.5: HF Space Crash Simulation
            print("\n[Test 3.5] Simulating HF Space Crash (sending audio, but HF space is unreachable)...")
            import subprocess
            # Kill the local HF space server running on port 8001
            subprocess.run(["pkill", "-f", "uvicorn app:app --host 0.0.0.0 --port 8001"], capture_output=True)
            
            # Resume healthy face, and start sending audio
            # Audio chunks will be sent, but HF space is dead, so call_hf_inference will fail
            # Timestamp won't update, so fusion_mode should go to face_only!
            for _ in range(6):
                await audio_ws.send(b"dummy_pcm")
                await video_ws.send(json.dumps({
                    "type": "landmarks",
                    "landmarks": [{"x":0, "y":0, "z":0}],
                    "face_confidence": 0.95
                }))
                await asyncio.sleep(0.5)
                
            # Read latest state
            while True:
                resp = json.loads(await video_ws.recv())
                if resp["fusion_mode"] == "face_only":
                    break
            print("✅ State 3.5 Passed: fusion_mode correctly transitioned to 'face_only' despite active audio transmission (graceful HTTP failure)")
            
            # Restart HF space server for Test 4
            print("Restarting HF Space Server...")
            subprocess.Popen(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"], cwd="../hf_space")
            await asyncio.sleep(2) # wait for it to boot
            
            # State 4: Recovery (Both back to normal)
            print("\n[Test 4] Resuming healthy HF Space Inference...")
            start_recovery = time.time()
            await audio_ws.send(b"dummy_pcm")
            await video_ws.send(json.dumps({
                "type": "landmarks",
                "landmarks": [{"x":0, "y":0, "z":0}],
                "face_confidence": 0.95
            }))
            
            # Because fusion loop runs every 0.5s, it should recover within 2 cycles
            while True:
                resp = json.loads(await video_ws.recv())
                if resp["fusion_mode"] == "full":
                    recovery_time = time.time() - start_recovery
                    print(f"✅ State 4 Passed: Recovered to 'full' in {recovery_time:.2f} seconds!")
                    break
                    
            print("\n🎉 All Graceful Degradation & Microservice Architecture tests passed successfully!")

    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_degradation())
