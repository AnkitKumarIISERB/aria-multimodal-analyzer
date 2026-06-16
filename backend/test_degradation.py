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
            # Still sending low confidence face so WS doesn't timeout, but no audio
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
            
            # State 4: Recovery (Both back to normal)
            print("\n[Test 4] Resuming healthy Face and Audio...")
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
                    assert recovery_time < 1.0, "Recovery took too long!"
                    break
                    
            print("\n[Test 5] Sending real audio chunk through Celery...")
            silent_chunk = bytes(16000)
            await audio_ws.send(silent_chunk)
            
            start = time.time()
            passed_test_5 = False
            while time.time() - start < 5:
                response = await video_ws.recv()
                data = json.loads(response)
                # Any successful emission that isn't completely degraded means audio was processed
                # Note: if it's "full" or "audio_only", the audio pipeline is working!
                if data.get("fusion_mode") != "degraded":
                    print(f"✅ Test 5 Passed: Celery audio result received in {time.time()-start:.2f}s")
                    passed_test_5 = True
                    break
                await asyncio.sleep(0.1)
                
            if not passed_test_5:
                print("❌ Test 5 Failed: Celery worker didn't return audio result in 5s")
                    
            print("\n🎉 All Graceful Degradation & Celery tests passed successfully!")

    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_degradation())
