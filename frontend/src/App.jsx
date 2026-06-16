import { useEffect, useRef, useState } from 'react'
import { motion, animate } from 'framer-motion'
import { Video, Mic, Zap, AlertTriangle, Play, Square, Activity } from 'lucide-react'
import './App.css'

// Helper component to animate numbers smoothly
function AnimatedNumber({ value }) {
  const nodeRef = useRef(null)

  useEffect(() => {
    const node = nodeRef.current
    if (!node || value === null) return
    const controls = animate(parseFloat(node.textContent) || 0, value, {
      duration: 0.5,
      ease: "easeOut",
      onUpdate(v) {
        node.textContent = v.toFixed(2)
      }
    })
    return () => controls.stop()
  }, [value])

  return <span ref={nodeRef}>{value === null ? "" : value.toFixed(2)}</span>
}

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
    <div className="app-container">
      <div className="header">
        <h1>Aria</h1>
        <p>Multimodal Emotion Fusion & Conflict Detection</p>
      </div>
      
      <div className="controls">
        {!sessionActive ? (
          <button className="btn btn-primary" onClick={startSession}>
            <Play size={20} /> Start Analysis
          </button>
        ) : (
          <button className="btn btn-danger" onClick={endSession}>
            <Square size={20} /> Stop Session
          </button>
        )}
      </div>

      <div className="dashboard-grid">
        {/* Left: Video Feed */}
        <div className="video-section">
          <motion.div 
            className={`video-container mode-${metrics?.fusion_mode || 'degraded'}`}
            layout
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <video 
              ref={videoRef} 
              style={{ display: 'none' }} 
              playsInline 
              onLoadedData={handleVideoLoaded}
            />
            <canvas ref={canvasRef} width={640} height={480} />
          </motion.div>
        </div>
        
        {/* Right: Metrics Dashboard */}
        {sessionActive && metrics && (
          <motion.div 
            className="glass-panel metrics-dashboard"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="status-header">
              <h2>System Status</h2>
              <span className={`badge ${metrics.fusion_mode}`}>
                {metrics.fusion_mode.replace('_', ' ')}
              </span>
            </div>
            
            <div className="metrics-grid">
              
              {/* Face Score */}
              <div className="metric-card metric-face">
                <div className="metric-info">
                  <div className="metric-icon"><Video size={20} /></div>
                  <div className="metric-name">Facial Emotion</div>
                </div>
                <div className="metric-value">
                  {metrics.face_score === null ? 
                    <span className="metric-null">LOW CONF</span> : 
                    <AnimatedNumber value={metrics.face_score} />
                  }
                </div>
              </div>
              
              {/* Audio Score */}
              <div className="metric-card metric-audio">
                <div className="metric-info">
                  <div className="metric-icon"><Mic size={20} /></div>
                  <div className="metric-name">Vocal Tone</div>
                </div>
                <div className="metric-value">
                  {metrics.audio_score === null ? 
                    <span className="metric-null">STALE</span> : 
                    <AnimatedNumber value={metrics.audio_score} />
                  }
                </div>
              </div>
              
              {/* Fused Score */}
              <div className="metric-card metric-fused">
                <div className="metric-info">
                  <div className="metric-icon"><Zap size={20} /></div>
                  <div className="metric-name">Fused Output</div>
                </div>
                <div className="metric-value">
                  <AnimatedNumber value={metrics.fused_score} />
                </div>
              </div>

              {/* Conflict Classifier */}
              <div className={`metric-card metric-conflict ${metrics.conflict_score > 0.5 ? 'conflict-active' : ''}`}>
                <div className="metric-info">
                  <div className="metric-icon">
                    {metrics.conflict_score > 0.5 ? <AlertTriangle size={20} /> : <Activity size={20} />}
                  </div>
                  <div className="metric-name">Conflict Detection</div>
                </div>
                <div className="metric-value">
                  {metrics.conflict_score === null ? 
                    <span className="metric-null">N/A</span> : 
                    <AnimatedNumber value={metrics.conflict_score * 100} /><span style={{fontSize: '1rem'}}>%</span>
                  }
                </div>
              </div>

            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default App
