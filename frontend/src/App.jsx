import { useEffect, useRef, useState } from 'react'
import { motion, animate } from 'framer-motion'
import { Video, Mic, Zap, AlertTriangle, Play, Square, Activity, Eye, EyeOff, Clock, HardDrive } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import './App.css'

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
  const xs = landmarks.map(l => l.x);
  const ys = landmarks.map(l => l.y);
  const boxArea = (Math.max(...xs) - Math.min(...xs)) *
    (Math.max(...ys) - Math.min(...ys));

  const yaw = Math.abs(landmarks[234].z - landmarks[454].z);
  const pitch = Math.abs(landmarks[10].z - landmarks[152].z);

  const sizeScore = Math.min(boxArea / 0.15, 1.0);
  const poseScore = Math.max(0, 1 - (yaw + pitch) / 0.3);

  return (sizeScore * 0.6) + (poseScore * 0.4);
}

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const waveformCanvasRef = useRef(null)
  const [sessionActive, setSessionActive] = useState(false)
  const [sessionId] = useState("session-" + Math.floor(Math.random() * 10000))
  const [metrics, setMetrics] = useState(null)

  // New State for UI expansions
  const [metricsHistory, setMetricsHistory] = useState([])
  const [showHud, setShowHud] = useState(false)
  const [sessionDuration, setSessionDuration] = useState(0)
  const [maxConflict, setMaxConflict] = useState(0)
  const [chunksProcessed, setChunksProcessed] = useState(0)

  const showHudRef = useRef(showHud)
  useEffect(() => { showHudRef.current = showHud }, [showHud])

  // WebSockets & Audio Context
  const videoWsRef = useRef(null)
  const audioWsRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const animationFrameRef = useRef(null)
  const timerRef = useRef(null)

  const drawWaveform = () => {
    if (!analyserRef.current || !waveformCanvasRef.current) return

    const canvas = waveformCanvasRef.current
    const ctx = canvas.getContext("2d")
    const bufferLength = analyserRef.current.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    analyserRef.current.getByteTimeDomainData(dataArray)

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = 2
    ctx.strokeStyle = '#00E5FF'
    ctx.beginPath()

    const sliceWidth = canvas.width * 1.0 / bufferLength
    let x = 0

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0
      const y = v * canvas.height / 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }

    ctx.lineTo(canvas.width, canvas.height / 2)
    ctx.stroke()

    animationFrameRef.current = requestAnimationFrame(drawWaveform)
  }

  const startSession = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      // Initialize AudioContext at 16kHz to match WavLM's expected sample rate
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      const source = audioCtxRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioCtxRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)
      drawWaveform()

      // Initialize WebSockets dynamically from VITE_API_URL
      const WS_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000")
        .replace(/^https/, "wss")
        .replace(/^http/, "ws");

      audioWsRef.current = new WebSocket(`${WS_BASE}/ws/stream/audio/${sessionId}`)
      videoWsRef.current = new WebSocket(`${WS_BASE}/ws/stream/video/${sessionId}`)

      videoWsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data)

        setMetrics(data)

        setMetricsHistory(prev => {
          const newHist = [...prev, {
            time: new Date().toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }),
            fused: data.fused_score || 0,
            conflict: data.conflict_score || 0
          }]
          if (newHist.length > 30) newHist.shift()
          return newHist
        })

        if (data.conflict_score > maxConflict) setMaxConflict(data.conflict_score)
        setChunksProcessed(prev => prev + 1)
      }

      document.addEventListener("visibilitychange", handleVisibilityChange)

      // Use ScriptProcessorNode to send raw PCM Float32 at 16kHz
      // Buffer size 8192 at 16kHz ≈ 512ms chunks, matching the fusion loop interval
      const processor = audioCtxRef.current.createScriptProcessor(8192, 1, 1)
      source.connect(processor)
      processor.connect(audioCtxRef.current.destination)
      processor.onaudioprocess = (e) => {
        const pcm = e.inputBuffer.getChannelData(0) // Float32Array
        if (audioWsRef.current?.readyState === WebSocket.OPEN) {
          audioWsRef.current.send(pcm.buffer)
        }
      }
      mediaRecorderRef.current = processor // store for cleanup

      setSessionActive(true)
      timerRef.current = setInterval(() => setSessionDuration(p => p + 1), 1000)
    } catch (err) {
      console.error("Error starting session:", err)
    }
  }

  const handleVisibilityChange = () => {
    if (document.hidden && videoWsRef.current?.readyState === WebSocket.OPEN) {
      videoWsRef.current.send(JSON.stringify({ type: "video_paused" }))
    }
  }

  const endSession = () => {
    if (mediaRecorderRef.current) mediaRecorderRef.current.disconnect()
    if (videoWsRef.current) videoWsRef.current.close()
    if (audioWsRef.current) audioWsRef.current.close()
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current)
    if (audioCtxRef.current) audioCtxRef.current.close()
    if (timerRef.current) clearInterval(timerRef.current)

    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(track => track.stop())
    }
    setSessionActive(false)
    setSessionDuration(0)
    setMetricsHistory([])
  }

  const faceMeshRef = useRef(null)
  const cameraRef = useRef(null)

  const handleVideoLoaded = () => {
    if (faceMeshRef.current) return

    const faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    })

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })

    faceMesh.onResults((results) => {
      const canvasCtx = canvasRef.current?.getContext('2d')
      if (canvasCtx && videoRef.current) {
        canvasCtx.save()
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height)

        // HUD Overlay Toggle
        if (showHudRef.current && results.multiFaceLandmarks) {
          for (const landmarks of results.multiFaceLandmarks) {
            canvasCtx.fillStyle = "rgba(0, 229, 255, 0.7)"
            for (const pt of landmarks) {
              const x = pt.x * canvasRef.current.width
              const y = pt.y * canvasRef.current.height
              canvasCtx.fillRect(x, y, 2, 2)
            }
          }
        }
        canvasCtx.restore()
      }

      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        if (videoWsRef.current?.readyState === WebSocket.OPEN) {
          const landmarks = results.multiFaceLandmarks[0];
          const face_confidence = computeFaceConfidence(landmarks, canvasRef.current.width, canvasRef.current.height);
          videoWsRef.current.send(JSON.stringify({ type: "landmarks", landmarks, face_confidence }))
        }
      }
    })

    faceMeshRef.current = faceMesh
    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current) await faceMesh.send({ image: videoRef.current })
      },
      width: 640,
      height: 480
    })
    camera.start()
    cameraRef.current = camera
  }

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0')
    const s = (seconds % 60).toString().padStart(2, '0')
    return `${m}:${s}`
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
          <>
            <button className="btn btn-secondary" onClick={() => setShowHud(!showHud)}>
              {showHud ? <EyeOff size={20} /> : <Eye size={20} />} {showHud ? "Hide HUD" : "Show Face HUD"}
            </button>
            <button className="btn btn-danger" onClick={endSession}>
              <Square size={20} /> Stop Session
            </button>
          </>
        )}
      </div>

      <div className="dashboard-grid">
        {/* Left: Video & Graph */}
        <div className="video-section">
          <motion.div
            className={`video-container mode-${metrics?.fusion_mode || 'degraded'}`}
            layout initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          >
            <video ref={videoRef} style={{ display: 'none' }} playsInline onLoadedData={handleVideoLoaded} />
            <canvas ref={canvasRef} width={640} height={480} />

            {sessionActive && (
              <div className="waveform-overlay">
                <canvas ref={waveformCanvasRef} width={640} height={80} />
              </div>
            )}
          </motion.div>

          {sessionActive && metricsHistory.length > 0 && (
            <motion.div className="glass-panel graph-panel" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="graph-header">Live Timeseries (30s)</div>
              <div style={{ width: '100%', height: 200 }}>
                <ResponsiveContainer>
                  <LineChart data={metricsHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="time" stroke="rgba(255,255,255,0.3)" fontSize={12} />
                    <YAxis stroke="rgba(255,255,255,0.3)" fontSize={12} domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                    <Line type="monotone" dataKey="fused" stroke="#FFFFFF" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="conflict" stroke="#FF1744" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
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

              <div className="metric-card metric-fused">
                <div className="metric-info">
                  <div className="metric-icon"><Zap size={20} /></div>
                  <div className="metric-name">Fused Output</div>
                </div>
                <div className="metric-value">
                  <AnimatedNumber value={metrics.fused_score} />
                </div>
              </div>

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
                    <><AnimatedNumber value={metrics.conflict_score * 100} /><span style={{ fontSize: '1rem' }}>%</span></>
                  }
                </div>
              </div>
            </div>

            <div className="session-stats">
              <div className="stat-item">
                <Clock size={16} /> {formatTime(sessionDuration)}
              </div>
              <div className="stat-item">
                <AlertTriangle size={16} /> Max Conflict: {(maxConflict * 100).toFixed(0)}%
              </div>
              <div className="stat-item">
                <HardDrive size={16} /> Data Chunks: {chunksProcessed}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default App
