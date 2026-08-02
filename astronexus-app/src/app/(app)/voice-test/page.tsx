'use client'
/**
 * /voice-test  — Isolated STT pipeline test page.
 *
 * Verifies ONLY:  🎤 Mic → MediaRecorder → POST /voice/transcribe → Whisper → Transcript
 *
 * No RAG, No LLM, No TTS, No Knowledge Graph, No Web Search.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FiMic, FiUpload, FiSquare,
  FiCheckCircle, FiLoader, FiWifi, FiAlertCircle,
} from 'react-icons/fi'
import { apiConfig } from '@/lib/config'

// ── Types ─────────────────────────────────────────────────────────────────────

type PermStatus = 'unknown' | 'checking' | 'granted' | 'denied'
type Stage =
  | 'idle'
  | 'recording'
  | 'recorded'
  | 'uploading'
  | 'transcribing'
  | 'done'
  | 'error'

interface TranscribeResponse {
  success:         boolean
  transcript:      string
  language:        string
  duration_sec:    number
  processing_time: number
  model:           string
  device:          string
  audio_format:    string
  audio_size_kb:   number
  segments:        { start: number; end: number; text: string }[]
  error?:          string
}

interface DebugInfo {
  backendReachable: boolean | null
  permStatus:       PermStatus
  audioFormat:      string
  sampleRate:       number | null
  audioSizeKb:      number | null
  whisperModel:     string
  language:         string
  processingTimeSec: number | null
  device:           string
}

// ── Waveform bar visualiser ───────────────────────────────────────────────────

function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef    = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analyser) return
    const ctx = canvas.getContext('2d')!

    const draw = () => {
      const buf = new Uint8Array(analyser.frequencyBinCount)
      analyser.getByteFrequencyData(buf)

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const bars = 36
      const W = canvas.width / bars

      for (let i = 0; i < bars; i++) {
        const idx = Math.floor((i / bars) * buf.length)
        const h   = (buf[idx] / 255) * canvas.height * 0.85 + 2
        const y   = (canvas.height - h) / 2
        ctx.fillStyle = `rgba(180,180,180,${0.3 + (buf[idx] / 255) * 0.7})`
        ctx.beginPath()
        ctx.roundRect(i * W + 2, y, W - 4, h, 3)
        ctx.fill()
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [analyser])

  return (
    <canvas ref={canvasRef} width={320} height={48}
      className="w-full rounded-xl opacity-90" />
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function VoiceTestPage() {
  const [stage, setStage]           = useState<Stage>('idle')
  const [elapsed, setElapsed]       = useState(0)
  const [transcript, setTranscript] = useState('')
  const [segments, setSegments]     = useState<TranscribeResponse['segments']>([])
  const [errorMsg, setErrorMsg]     = useState('')
  const [blobUrl, setBlobUrl]       = useState<string | null>(null)
  const [analyser, setAnalyser]     = useState<AnalyserNode | null>(null)
  const [debug, setDebug]           = useState<DebugInfo>({
    backendReachable:  null,
    permStatus:        'unknown',
    audioFormat:       '—',
    sampleRate:        null,
    audioSizeKb:       null,
    whisperModel:      'base',
    language:          '—',
    processingTimeSec: null,
    device:            '—',
  })

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef        = useRef<Blob[]>([])
  const blobRef          = useRef<Blob | null>(null)
  const timerRef         = useRef<ReturnType<typeof setInterval> | null>(null)
  const streamRef        = useRef<MediaStream | null>(null)
  const audioCtxRef      = useRef<AudioContext | null>(null)

  // ── Backend health check ──────────────────────────────────────────────────

  useEffect(() => {
    fetch(`${apiConfig.baseUrl}/health`)
      .then(r => r.ok
        ? setDebug(d => ({ ...d, backendReachable: true }))
        : setDebug(d => ({ ...d, backendReachable: false }))
      )
      .catch(() => setDebug(d => ({ ...d, backendReachable: false })))
  }, [])

  // ── Mic permission check ──────────────────────────────────────────────────

  const checkPermission = useCallback(async () => {
    setDebug(d => ({ ...d, permStatus: 'checking' }))
    try {
      const perm = await navigator.permissions.query({ name: 'microphone' as PermissionName })
      setDebug(d => ({
        ...d,
        permStatus: perm.state === 'granted' ? 'granted'
                  : perm.state === 'denied'  ? 'denied'
                  : 'unknown',
      }))
    } catch {
      setDebug(d => ({ ...d, permStatus: 'unknown' }))
    }
  }, [])

  useEffect(() => { checkPermission() }, [checkPermission])

  // ── Recording ─────────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setErrorMsg('')
    setTranscript('')
    setSegments([])
    setBlobUrl(null)

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      streamRef.current = stream
      setDebug(d => ({ ...d, permStatus: 'granted', sampleRate: stream.getAudioTracks()[0].getSettings().sampleRate ?? null }))
    } catch (e) {
      const msg = e instanceof Error && e.name === 'NotAllowedError'
        ? 'Microphone access denied. Allow it in browser settings.'
        : `Microphone error: ${e instanceof Error ? e.message : e}`
      setErrorMsg(msg)
      setDebug(d => ({ ...d, permStatus: 'denied' }))
      setStage('error')
      return
    }

    // Visualiser
    try {
      const ctx  = new AudioContext()
      audioCtxRef.current = ctx
      const src  = ctx.createMediaStreamSource(stream)
      const an   = ctx.createAnalyser()
      an.fftSize = 64
      src.connect(an)
      setAnalyser(an)
    } catch { /* non-fatal */ }

    // MediaRecorder — prefer webm/opus
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : ''
    const fmt = mimeType.includes('mp4') ? 'mp4' : mimeType ? 'webm' : 'webm'
    setDebug(d => ({ ...d, audioFormat: fmt.toUpperCase() }))

    chunksRef.current = []
    const recorder    = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    mediaRecorderRef.current = recorder

    recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    recorder.onstop          = () => {
      stream.getTracks().forEach(t => t.stop())
      audioCtxRef.current?.close()
      setAnalyser(null)
      const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' })
      blobRef.current = blob
      const kb = blob.size / 1024
      setDebug(d => ({ ...d, audioSizeKb: Math.round(kb * 10) / 10 }))
      setBlobUrl(URL.createObjectURL(blob))
      setStage('recorded')
    }

    recorder.start(200)
    setStage('recording')
    setElapsed(0)

    timerRef.current = setInterval(() => setElapsed(prev => prev + 1), 1000)
  }, [])

  const stopRecording = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
    setStage('recorded')  // recorder.onstop will confirm
  }, [])

  // ── Upload + transcribe ───────────────────────────────────────────────────

  const upload = useCallback(async () => {
    const blob = blobRef.current
    if (!blob) { setErrorMsg('No audio recorded.'); setStage('error'); return }

    const fmt = debug.audioFormat.toLowerCase()
    const form = new FormData()
    form.append('audio', blob, `voice_test.${fmt}`)

    setStage('uploading')
    await new Promise(r => setTimeout(r, 300))  // let UI update

    setStage('transcribing')
    try {
      const res = await fetch(`${apiConfig.baseUrl}/voice/transcribe`, {
        method: 'POST',
        body:   form,
      })

      if (!res.ok) {
        let detail = `Backend returned ${res.status}`
        try { detail = (await res.json()).detail ?? detail } catch { /**/ }
        throw new Error(detail)
      }

      const data: TranscribeResponse = await res.json()

      if (!data.success) throw new Error(data.error || 'Transcription failed')

      setTranscript(data.transcript)
      setSegments(data.segments ?? [])
      setDebug(d => ({
        ...d,
        whisperModel:      data.model,
        language:          data.language,
        processingTimeSec: data.processing_time,
        device:            data.device,
        audioSizeKb:       data.audio_size_kb,
        audioFormat:       data.audio_format,
      }))
      setStage('done')
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e))
      setStage('error')
    }
  }, [debug.audioFormat])

  // ── Helpers ───────────────────────────────────────────────────────────────

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach(t => t.stop())
    audioCtxRef.current?.close()
    setAnalyser(null)
    setStage('idle')
    setElapsed(0)
    setTranscript('')
    setSegments([])
    setErrorMsg('')
    setBlobUrl(null)
    blobRef.current = null
    setDebug(d => ({ ...d, audioSizeKb: null, processingTimeSec: null, language: '—', device: '—' }))
  }

  const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  // ── Render ────────────────────────────────────────────────────────────────

  const isBusy = stage === 'uploading' || stage === 'transcribing'

  return (
    <div className="mx-auto max-w-2xl p-6 md:p-10">

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-2 text-xs text-dim mb-3">
          <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1">
            🧪 Test mode — isolated STT only
          </span>
        </div>
        <h1 className="font-display text-3xl font-bold text-light">🎤 Voice Test</h1>
        <p className="mt-1 text-sm text-dim">Verify microphone → Whisper STT pipeline with no other services.</p>
      </div>

      <div className="grid gap-5">

        {/* ── Recorder card ───────────────────────────────────────────── */}
        <div className="anx-card p-6">
          <div className="flex flex-col items-center gap-6">

            {/* Status label */}
            <AnimatePresence mode="wait">
              <motion.p key={stage} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="text-sm font-medium text-dim h-5">
                {stage === 'idle'         && 'Ready to record'}
                {stage === 'recording'    && '● Recording…'}
                {stage === 'recorded'     && 'Recording complete'}
                {stage === 'uploading'    && 'Uploading audio…'}
                {stage === 'transcribing' && 'Processing with Whisper…'}
                {stage === 'done'         && '✓ Transcription complete'}
                {stage === 'error'        && '⚠ Error'}
              </motion.p>
            </AnimatePresence>

            {/* Timer */}
            {stage === 'recording' && (
              <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }}
                className="font-mono text-4xl font-light text-light tabular-nums">
                {fmt(elapsed)}
              </motion.div>
            )}

            {/* Waveform */}
            {stage === 'recording' && analyser && (
              <div className="w-full">
                <Waveform analyser={analyser} />
              </div>
            )}

            {/* Spinner while processing */}
            {isBusy && (
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                <FiLoader className="h-8 w-8 text-silver" />
              </motion.div>
            )}

            {/* Playback */}
            {blobUrl && stage !== 'recording' && !isBusy && (
              <audio controls src={blobUrl}
                className="w-full h-10 accent-silver-light opacity-80 hover:opacity-100 transition-opacity"
                style={{ filter: 'grayscale(1) brightness(1.4)' }} />
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-3">
              {stage === 'idle' && (
                <button onClick={startRecording}
                  className="anx-button flex items-center gap-2.5 px-8 py-3.5 text-sm font-medium">
                  <FiMic className="h-4 w-4" /> Start Recording
                </button>
              )}
              {stage === 'recording' && (
                <button onClick={stopRecording}
                  className="flex items-center gap-2.5 rounded-[16px] border border-red-500/30 bg-red-500/10 px-8 py-3.5 text-sm font-medium text-red-300 transition-colors hover:bg-red-500/18">
                  <FiSquare className="h-4 w-4" /> Stop Recording
                </button>
              )}
              {stage === 'recorded' && (
                <>
                  <button onClick={upload}
                    className="anx-button flex items-center gap-2.5 px-8 py-3.5 text-sm font-medium">
                    <FiUpload className="h-4 w-4" /> Transcribe
                  </button>
                  <button onClick={reset}
                    className="rounded-[16px] border border-white/[0.10] bg-white/[0.04] px-5 py-3.5 text-sm text-dim transition-colors hover:text-light">
                    Re-record
                  </button>
                </>
              )}
              {(stage === 'done' || stage === 'error') && (
                <button onClick={reset}
                  className="anx-button flex items-center gap-2.5 px-8 py-3.5 text-sm font-medium">
                  <FiMic className="h-4 w-4" /> Record again
                </button>
              )}
            </div>

            {/* Error */}
            <AnimatePresence>
              {errorMsg && (
                <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="flex w-full items-start gap-2.5 rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3" role="alert">
                  <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
                  <p className="text-sm text-red-300">{errorMsg}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* ── Transcript ─────────────────────────────────────────────── */}
        <AnimatePresence>
          {stage === 'done' && transcript && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="anx-card p-6">
              <div className="mb-3 flex items-center gap-2">
                <FiCheckCircle className="h-4 w-4 text-silver-bright" />
                <span className="text-xs font-semibold uppercase tracking-wider text-dim">Transcript</span>
              </div>
              <p className="text-base leading-relaxed text-light">{transcript}</p>

              {/* Segments (optional detail) */}
              {segments.length > 1 && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-xs text-dim hover:text-silver-light">
                    Show {segments.length} word segments
                  </summary>
                  <div className="mt-3 space-y-1.5">
                    {segments.map((s, i) => (
                      <div key={i} className="flex items-baseline gap-3">
                        <span className="font-mono text-[10px] text-faint w-20 flex-shrink-0">
                          {s.start.toFixed(1)}s — {s.end.toFixed(1)}s
                        </span>
                        <span className="text-xs text-dim">{s.text}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Debug panel ────────────────────────────────────────────── */}
        <div className="anx-card p-5">
          <p className="mb-4 text-xs font-semibold uppercase tracking-wider text-dim">Debug Panel</p>
          <div className="grid gap-2.5">
            <DebugRow
              icon={<FiWifi className="h-3.5 w-3.5" />}
              label="Backend"
              value={
                debug.backendReachable === null ? 'Checking…'
                : debug.backendReachable        ? '✅ Connected'
                :                                 '❌ Unreachable'
              }
              ok={debug.backendReachable ?? undefined}
            />
            <DebugRow
              icon={<FiMic className="h-3.5 w-3.5" />}
              label="Microphone"
              value={
                debug.permStatus === 'granted' ? '✅ Granted'
                : debug.permStatus === 'denied'  ? '❌ Denied'
                : debug.permStatus === 'checking' ? 'Checking…'
                :                                   '❓ Unknown'
              }
              ok={debug.permStatus === 'granted' ? true : debug.permStatus === 'denied' ? false : undefined}
            />
            <DebugRow label="Audio Format"    value={debug.audioFormat} />
            <DebugRow label="Sample Rate"     value={debug.sampleRate ? `${debug.sampleRate.toLocaleString()} Hz` : '—'} />
            <DebugRow label="Audio Size"      value={debug.audioSizeKb !== null ? `${debug.audioSizeKb} KB` : '—'} />
            <DebugRow label="Whisper Model"   value={debug.whisperModel} />
            <DebugRow label="Device"          value={debug.device === 'cuda' ? '🖥 GPU (CUDA)' : debug.device === 'cpu' ? '💻 CPU' : debug.device} />
            <DebugRow label="Language"        value={debug.language === 'en' ? 'English' : debug.language} />
            <DebugRow label="Processing Time" value={debug.processingTimeSec !== null ? `${debug.processingTimeSec.toFixed(2)} sec` : '—'} />
          </div>
        </div>

      </div>
    </div>
  )
}

// ── DebugRow ─────────────────────────────────────────────────────────────────

function DebugRow({
  label, value, ok, icon,
}: { label: string; value: string; ok?: boolean; icon?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-white/[0.04] bg-white/[0.02] px-3.5 py-2.5">
      <div className="flex items-center gap-2 text-xs text-dim">
        {icon}
        <span>{label}</span>
      </div>
      <span className={`text-xs font-mono ${
        ok === true  ? 'text-silver-bright' :
        ok === false ? 'text-red-400'        :
        'text-silver'
      }`}>
        {value}
      </span>
    </div>
  )
}
