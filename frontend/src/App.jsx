import { useEffect, useRef, useState } from 'react'
import './App.css'

const FaceMesh = window.FaceMesh
const Camera = window.Camera

function computeFaceConfidence(landmarks, videoWidth, videoHeight) {
  // Bounding box coverage
  const xs = landmarks.map(l => l.x);
  const ys = landmarks.map(l => l.y);
  const boxArea = (Math.max(...xs) - Math.min(...xs)) * 
                  (Math.max(...ys) - Math.min(...ys));
  
  // Yaw estimation: difference in z between left/right temples
  // Landmarks 234 (right temple) and 454 (left temple)
  const yaw = Math.abs(landmarks[234].z - landmarks[454].z);
  
  // Pitch estimation: z difference between forehead and chin
  // Landmarks 10 (forehead) and 152 (chin)  
  const pitch = Math.abs(landmarks[10].z - landmarks[152].z);
  
  const sizeScore = Math.min(boxArea / 0.15, 1.0); // 0.15 = ~40% of frame
  const poseScore = Math.max(0, 1 - (yaw + pitch) / 0.3);
  
  return (sizeScore * 0.6) + (poseScore * 0.4);
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [sessionActive, setSessionActive] = useState(false)
  const [sessionId] = useState("session-" + Math.floor(Math.random() * 10000))
  const [metrics, setMetrics] = useState(null)
  
  // WebSockets
  const videoWsRef = useRef(null)
  const audioWsRef = useRef(null)
  
  // Media Recorder for audio
  const mediaRecorderRef = useRef(null)

  const startSession = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      
      // Initialize Audio WS
      audioWsRef.current = new WebSocket(`ws://localhost:8000/ws/stream/audio/${sessionId}`)
      audioWsRef.current.onopen = () => console.log("[Audio WS] Connected")
      
      // Initialize Video WS
      videoWsRef.current = new WebSocket(`ws://localhost:8000/ws/stream/video/${sessionId}`)
      videoWsRef.current.onopen = () => console.log("[Video WS] Connected")
      videoWsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)
        // console.log("--> [Server Fused State]:", data)
        setMetrics({
          face_score: data.face_score,
          audio_score: data.audio_score,
          fused_score: data.fused_score,
          conflict_score: data.conflict_score,
          fusion_mode: data.fusion_mode,
          face_stale: data.face_stale,
          audio_stale: data.audio_stale
        })
      }

      // Handle Page Visibility for stale data protection
      const handleVisibilityChange = () => {
        if (document.hidden && videoWsRef.current?.readyState === WebSocket.OPEN) {
          videoWsRef.current.send(JSON.stringify({ type: "video_paused" }))
        }
      }
      document.addEventListener("visibilitychange", handleVisibilityChange)

      // Start Audio recording (500ms chunks)
      // Note: In real ML pipeline we send PCM. For this skeleton, we just send dummy blobs.
      mediaRecorderRef.current = new MediaRecorder(stream)
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0 && audioWsRef.current?.readyState === WebSocket.OPEN) {
          // Sending dummy blob for now
          audioWsRef.current.send(event.data)
        }
      }
      mediaRecorderRef.current.start(500) // 500ms chunks
      
      setSessionActive(true)
    } catch (err) {
      console.error("Error starting session:", err)
    }
  }
  
  const endSession = () => {
    if (mediaRecorderRef.current) mediaRecorderRef.current.stop()
    if (videoWsRef.current) videoWsRef.current.close()
    if (audioWsRef.current) audioWsRef.current.close()
    
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(track => track.stop())
    }
    setSessionActive(false)
  }

  const faceMeshRef = useRef(null)
  const cameraRef = useRef(null)

  // Handle MediaPipe Initialization once video starts playing
  const handleVideoLoaded = () => {
    if (faceMeshRef.current) {
      console.log("MediaPipe already initialized, skipping.")
      return
    }
    console.log("Video loaded. Initializing MediaPipe FaceMesh...")
    
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      }
    })
    
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    
    faceMesh.onResults((results) => {
      // 1. Render to canvas
      const canvasCtx = canvasRef.current?.getContext('2d')
      if (canvasCtx && videoRef.current) {
        canvasCtx.save()
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height)
        
        // Render landmarks (dummy rendering for skeleton)
        if (results.multiFaceLandmarks) {
          for (const landmarks of results.multiFaceLandmarks) {
            canvasCtx.fillStyle = "#32CD32"
            for (const pt of landmarks) {
              const x = pt.x * canvasRef.current.width
              const y = pt.y * canvasRef.current.height
              canvasCtx.fillRect(x, y, 2, 2)
            }
          }
        }
        canvasCtx.restore()
      }
      
      // 2. Send landmarks to server over WS
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        if (videoWsRef.current?.readyState === WebSocket.OPEN) {
          const landmarks = results.multiFaceLandmarks[0];
          const face_confidence = computeFaceConfidence(landmarks, canvasRef.current.width, canvasRef.current.height);
          
          const payload = {
            type: "landmarks",
            landmarks: landmarks,
            face_confidence: face_confidence
          }
          // Sending REAL landmarks and confidence
          videoWsRef.current.send(JSON.stringify(payload))
        }
      }
    })
    
    faceMeshRef.current = faceMesh

    // Start camera loop
    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current) {
          await faceMesh.send({ image: videoRef.current })
        }
      },
      width: 640,
      height: 480
    })
    
    camera.start()
    cameraRef.current = camera
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px', padding: '40px' }}>
      <h1>Aria — Minimal WebSockets + MediaPipe Skeleton</h1>
      
      <div style={{ display: 'flex', gap: '10px' }}>
        {!sessionActive ? (
          <button onClick={startSession} style={{ padding: '10px 20px', fontSize: '16px' }}>Start Session</button>
        ) : (
          <button onClick={endSession} style={{ padding: '10px 20px', fontSize: '16px', background: 'red', color: 'white' }}>End Session</button>
        )}
      </div>

      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
        {/* Hidden video element used strictly for feeding MediaPipe */}
        <video 
          ref={videoRef} 
          style={{ display: 'none' }} 
          playsInline 
          onLoadedData={handleVideoLoaded}
        />
        
        {/* Visible canvas showing the webcam and landmarks */}
        <canvas 
          ref={canvasRef} 
          width={640} 
          height={480} 
          style={{ border: '2px solid #ccc', borderRadius: '8px', backgroundColor: '#000' }}
        />
      </div>
      
      {sessionActive && metrics && (
        <div style={{ 
          marginTop: '20px', 
          padding: '20px', 
          background: '#1a1a1a', 
          borderRadius: '12px', 
          color: 'white',
          width: '640px',
          boxSizing: 'border-box'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #333', paddingBottom: '10px', marginBottom: '15px' }}>
            <h2 style={{ margin: 0 }}>Live ML Diagnostics</h2>
            <div style={{
              background: metrics.fusion_mode === 'full' ? '#32CD32' : 
                          metrics.fusion_mode === 'degraded' ? '#ff4444' : '#f39c12',
              color: 'black', padding: '4px 10px', borderRadius: '12px', fontSize: '12px', fontWeight: 'bold'
            }}>
              MODE: {metrics.fusion_mode?.toUpperCase()}
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <div style={{ background: '#333', padding: '15px', borderRadius: '8px' }}>
              <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '5px' }}>Face Emotion (MediaPipe)</div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: metrics.face_score === null ? '#ff4444' : '#32CD32' }}>
                {metrics.face_score === null ? 'NULL (Low Conf)' : metrics.face_score.toFixed(2)}
              </div>
            </div>
            
            <div style={{ background: '#333', padding: '15px', borderRadius: '8px' }}>
              <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '5px' }}>Audio Emotion (WavLM)</div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: metrics.audio_score === null ? '#ff4444' : '#00BFFF' }}>
                {metrics.audio_score === null ? 'NULL (Stale)' : metrics.audio_score.toFixed(2)}
              </div>
            </div>
            
            <div style={{ background: '#333', padding: '15px', borderRadius: '8px', border: '1px solid #00BFFF' }}>
              <div style={{ fontSize: '14px', color: '#00BFFF', marginBottom: '5px' }}>Fused Score</div>
              <div style={{ fontSize: '28px', fontWeight: 'bold', color: 'white' }}>
                {(metrics.fused_score || 0).toFixed(2)}
              </div>
            </div>

            <div style={{ background: '#333', padding: '15px', borderRadius: '8px', border: metrics.conflict_score > 0.5 ? '2px solid #ff4444' : '1px solid #555' }}>
              <div style={{ fontSize: '14px', color: metrics.conflict_score > 0.5 ? '#ff4444' : '#aaa', marginBottom: '5px' }}>Conflict Classifier (LR)</div>
              <div style={{ fontSize: '28px', fontWeight: 'bold', color: metrics.conflict_score > 0.5 ? '#ff4444' : (metrics.conflict_score === null ? '#aaa' : 'white') }}>
                {metrics.conflict_score === null ? 'N/A' : `${(metrics.conflict_score * 100).toFixed(1)}%`}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
