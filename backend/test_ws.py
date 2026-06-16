import asyncio
import websockets
import json
import time

async def test_audio_ws(session_id):
    uri = f"ws://localhost:8000/ws/stream/audio/{session_id}"
    try:
        async with websockets.connect(uri) as websocket:
            print("[Test Audio WS] Connected")
            for i in range(10): # Send 10 chunks, 2 per second (500ms intervals)
                dummy_pcm = b'\x00\x01' * 512
                await websocket.send(dummy_pcm)
                print(f"[Test Audio WS] Sent chunk {i+1}")
                await asyncio.sleep(0.5)
    except Exception as e:
        print(f"[Test Audio WS] Error: {e}")

async def test_video_ws(session_id):
    uri = f"ws://localhost:8000/ws/stream/video/{session_id}"
    try:
        async with websockets.connect(uri) as websocket:
            print("[Test Video WS] Connected")
            
            # Start listener task for incoming fusion responses
            async def listen():
                try:
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        print(f"--> [Test Video WS] Received fused state: {data}")
                except Exception as e:
                    print(f"[Test Video WS] Listener error: {e}")
            
            listener_task = asyncio.create_task(listen())
            
            # Send landmark vectors at 10fps for 5 seconds
            for i in range(50):
                dummy_landmarks = {
                    "type": "landmarks",
                    "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}] * 468
                }
                await websocket.send(json.dumps(dummy_landmarks))
                # print(f"[Test Video WS] Sent landmark {i+1}")
                await asyncio.sleep(0.1) # 100ms
            
            # Test visibility pause
            print("[Test Video WS] Simulating tab hidden...")
            await websocket.send(json.dumps({"type": "video_paused"}))
            await asyncio.sleep(3.0) # Wait 3 seconds, should see stale payload
            
            listener_task.cancel()
            
    except Exception as e:
        print(f"[Test Video WS] Error: {e}")

async def main():
    session_id = "test-session-123"
    print(f"Starting test for session: {session_id}")
    
    # Run both connections simultaneously
    await asyncio.gather(
        test_audio_ws(session_id),
        test_video_ws(session_id)
    )

if __name__ == "__main__":
    asyncio.run(main())
