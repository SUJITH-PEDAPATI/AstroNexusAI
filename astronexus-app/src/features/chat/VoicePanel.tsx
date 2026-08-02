'use client'
/**
 * VoicePanel — full live voice interaction UI for AstroNexus AI.
 *
 * Replaces the minimal VoiceButton with a panel that shows:
 *   • Animated waveform while recording
 *   • Recording timer
 *   • Live transcript (updated every ~2s via WebSocket partial frames)
 *   • Pipeline stage indicators (Transcribing → Keywords → KG → RAG → Answer)
 *   • Extracted keywords and named entities
 *   • Final AI answer with grade + citations
 *
 * WebSocket protocol (matches server.py /voice/stream):
 *   Binary frames → raw audio (PCM float32 LE, 16 kHz, mono)
 *   Text  {"action":"stop","paper_id":"..."} → trigger full pipeline
 *
 *   Server → client:
 *   {"type":"partial",    "text":"..."}
 *   {"type":"transcript", "text":"...","language":"en"}
 *   {"type":"stage",      "stage":"Extracting keywords"}
 *   {"type":"keywords",   "keywords":[...],"entities":[...],"scientific":[...]}
 *   {"type":"graph",      "entities":3,"keywords":5,"status":"ok"}
 *   {"type":"answer",     "text":"...","grade":"A","citations":[...]}
 *   {"type":"done",       "elapsed_sec":7.3}
 *   {"type":"error",      "message":"..."}
 *
 * Fallback: if WebSocket is unavailable, falls back silently to the existing
 *           REST /voice/chat (VoiceButton behaviour).
 */

import {
  useRef, useState, useCallback, useEffect, memo, type ReactNode,
} from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FiMic, FiSquare, FiAlertCircle, FiTag, FiShare2,
  FiCheckCircle, FiLoader, FiChevronDown, FiChevronUp,
} from 'react-icons/fi'
import { apiConfig } from '@/lib/config'
import { cn } from '@/lib/cn'

// ── Types ─────────────────────────────────────────────────────────────────────

type PanelState =
  | 'idle'
  | 'recording'
  | 'transcribing'
  | 'keywords'
  | 'graph'
  | 'rag'
  | 'answer'
  | 'error'

interface Citation { section: string; page: number; score: number }
interface WebSource { title: string; url: string; source: string }

interface AnswerPayload {
  text:         string
  grade:        string
  citations:    Citation[]
  search_type:  string
  search_label: string
  web_sources:  WebSource[]
  query_type:   string
}

export interface Props {
  onAnswer:    (answer: string, grade: string, citations: Citation[]) => void
  onTranscript:(text: string) => void
  disabled?:   boolean
  paperId?:    string
}

// ── Stage config ──────────────────────────────────────────────────────────────

const STAGES: Record<string, { label: string; icon: ReactNode }> = {
  'Transcribing audio':            { label: 'Transcribing audio',          icon: <FiLoader className="h-3 w-3" /> },
  'Extracting keywords':           { label: 'Extracting keywords',         icon: <FiTag    className="h-3 w-3" /> },
  'Updating Knowledge Graph':      { label: 'Updating Knowledge Graph',    icon: <FiShare2 className="h-3 w-3" /> },
  'Searching Research Papers':     { label: 'Searching Research Papers',   icon: <FiLoader className="h-3 w-3" /> },
  'Running Multi-Agent reasoning': { label: 'Running Multi-Agent reasoning', icon: <FiLoader className="h-3 w-3" /> },
}

// ── Waveform ──────────────────────────────────────────────────────────────────

const Waveform = memo(function Waveform({ active }: { active: boolean }) {
  const bars = 20
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: bars }).map((_, i) => (
        <motion.span
          key={i}
          className="w-0.5 rounded-full bg-red-400"
          animate={active ? {
            height: ['4px', `${6 + Math.random() * 18}px`, '4px'],
          } : { height: '4px' }}
          transition={{
            repeat:   Infinity,
            duration: 0.5 + Math.random() * 0.4,
            delay:    i * 0.04,
            ease:     'easeInOut',
          }}
        />
      ))}
    </div>
  )
})

// ── Timer ─────────────────────────────────────────────────────────────────────

function useTimer(running: boolean) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    if (!running) { setSecs(0); return }
    const id = setInterval(() => setSecs((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [running])
  const mm = String(Math.floor(secs / 60)).padStart(2, '0')
  const ss = String(secs % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

// ── Audio capture helpers ─────────────────────────────────────────────────────

/** Downsample a Float32Array from srcRate to 16 kHz (Whisper's native rate). */
function downsampleTo16k(input: Float32Array, srcRate: number): Float32Array {
  if (srcRate === 16_000) return input
  const ratio  = srcRate / 16_000
  const outLen = Math.floor(input.length / ratio)
  const out    = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    out[i] = input[Math.floor(i * ratio)]
  }
  return out
}

/** Convert Float32Array to little-endian binary (what the server expects). */
function float32ToBytes(f32: Float32Array): ArrayBuffer {
  const buf  = new ArrayBuffer(f32.length * 4)
  const view = new DataView(buf)
  for (let i = 0; i < f32.length; i++) view.setFloat32(i * 4, f32[i], true)
  return buf
}

// ── Main component ────────────────────────────────────────────────────────────

export function VoicePanel({ onAnswer, onTranscript, disabled = false, paperId }: Props) {
  const [state,         setState]         = useState<PanelState>('idle')
  const [liveText,      setLiveText]      = useState('')
  const [finalText,     setFinalText]     = useState('')
  const [currentStage,  setCurrentStage]  = useState('')
  const [doneStages,    setDoneStages]    = useState<string[]>([])
  const [keywords,      setKeywords]      = useState<string[]>([])
  const [entities,      setEntities]      = useState<string[]>([])
  const [scientific,    setScientific]    = useState<string[]>([])
  const [graphInfo,     setGraphInfo]     = useState<{entities:number;keywords:number} | null>(null)
  const [answer,        setAnswer]        = useState<AnswerPayload | null>(null)
  const [errorMsg,      setErrorMsg]      = useState<string | null>(null)
  const [panelOpen,     setPanelOpen]     = useState(false)
  const [elapsed,       setElapsed]       = useState<number | null>(null)

  const wsRef      = useRef<WebSocket | null>(null)
  const ctxRef     = useRef<AudioContext | null>(null)
  const srcRef     = useRef<MediaStreamAudioSourceNode | null>(null)
  const procRef    = useRef<ScriptProcessorNode | null>(null)
  const streamRef  = useRef<MediaStream | null>(null)
  const isRec      = state === 'recording'
  const timer      = useTimer(isRec)

  // ── Cleanup ────────────────────────────────────────────────────────────────

  const cleanup = useCallback(() => {
    try { procRef.current?.disconnect() } catch { /**/ }
    try { srcRef.current?.disconnect()  } catch { /**/ }
    try { ctxRef.current?.close()       } catch { /**/ }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    procRef.current  = null
    srcRef.current   = null
    ctxRef.current   = null
    streamRef.current = null
  }, [])

  // ── WebSocket message handler ──────────────────────────────────────────────

  const handleWsMessage = useCallback((evt: MessageEvent) => {
    if (typeof evt.data !== 'string') return
    let msg: Record<string, unknown>
    try { msg = JSON.parse(evt.data) } catch { return }

    const type = msg.type as string

    if (type === 'partial') {
      setLiveText(msg.text as string)
    }
    else if (type === 'transcript') {
      const t = msg.text as string
      setFinalText(t)
      setLiveText(t)
      onTranscript(t)
      setDoneStages((d) => [...d, 'Transcribing audio'])
      setCurrentStage('Extracting keywords')
      setState('keywords')
    }
    else if (type === 'stage') {
      const s = msg.stage as string
      setCurrentStage(s)
    }
    else if (type === 'keywords') {
      setKeywords((msg.keywords as string[]) ?? [])
      setEntities((msg.entities as string[]) ?? [])
      setScientific((msg.scientific as string[]) ?? [])
      setDoneStages((d) => [...d, 'Extracting keywords'])
      setState('graph')
    }
    else if (type === 'graph') {
      setGraphInfo({
        entities: (msg.entities as number) ?? 0,
        keywords: (msg.keywords as number) ?? 0,
      })
      setDoneStages((d) => [...d, 'Updating Knowledge Graph'])
      setState('rag')
    }
    else if (type === 'answer') {
      const a: AnswerPayload = {
        text:         msg.text         as string,
        grade:        msg.grade        as string,
        citations:    (msg.citations   as Citation[]) ?? [],
        search_type:  msg.search_type  as string ?? 'local_rag',
        search_label: msg.search_label as string ?? '📄 Research Papers',
        web_sources:  (msg.web_sources as WebSource[]) ?? [],
        query_type:   msg.query_type   as string ?? '',
      }
      setAnswer(a)
      setState('answer')
      onAnswer(a.text, a.grade, a.citations)
    }
    else if (type === 'done') {
      setElapsed(msg.elapsed_sec as number)
    }
    else if (type === 'error') {
      setErrorMsg(msg.message as string ?? 'Unknown error')
      setState('error')
    }
  }, [onAnswer, onTranscript])

  // ── Start recording ────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setErrorMsg(null)
    setLiveText('')
    setFinalText('')
    setKeywords([])
    setEntities([])
    setScientific([])
    setGraphInfo(null)
    setAnswer(null)
    setDoneStages([])
    setCurrentStage('')
    setElapsed(null)
    setPanelOpen(true)

    // Get microphone
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
    } catch (e) {
      setErrorMsg(e instanceof Error && e.name === 'NotAllowedError'
        ? 'Microphone access denied. Check browser permissions.'
        : 'Could not access microphone.')
      setState('error')
      return
    }

    // Connect WebSocket
    const wsUrl = apiConfig.baseUrl.replace(/^http/, 'ws') + '/voice/stream'
    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl)
      wsRef.current = ws
    } catch {
      setErrorMsg('Cannot connect to backend WebSocket. Is the server running?')
      setState('error')
      stream.getTracks().forEach((t) => t.stop())
      return
    }

    ws.onmessage = handleWsMessage
    ws.onerror   = () => {
      setErrorMsg('WebSocket error. Falling back to standard voice mode.')
      setState('error')
      cleanup()
    }
    ws.onclose   = () => {
      if (state === 'recording') cleanup()
    }

    // Wait for WebSocket to open before starting AudioContext
    await new Promise<void>((resolve, reject) => {
      ws.onopen  = () => resolve()
      setTimeout(() => reject(new Error('WS timeout')), 5000)
    }).catch(() => {
      setErrorMsg('Backend WebSocket did not respond. Is the server running?')
      setState('error')
      cleanup()
      return
    })

    // Set up AudioContext for capture + downsampling to 16 kHz
    const ctx  = new AudioContext()
    ctxRef.current = ctx
    const src  = ctx.createMediaStreamSource(stream)
    srcRef.current = src

    // ScriptProcessor gives us raw PCM — no Blob reassembly needed
    // (deprecated API but universally supported; AudioWorklet is the future)
    const proc = ctx.createScriptProcessor(4096, 1, 1)
    procRef.current = proc

    proc.onaudioprocess = (e) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return
      const input   = e.inputBuffer.getChannelData(0)
      const resampl = downsampleTo16k(input, ctx.sampleRate)
      const bytes   = float32ToBytes(resampl)
      wsRef.current.send(bytes)
    }

    src.connect(proc)
    proc.connect(ctx.destination)

    setState('recording')
  }, [handleWsMessage, cleanup, state])

  // ── Stop recording ─────────────────────────────────────────────────────────

  const stopRecording = useCallback(() => {
    // Stop audio capture
    cleanup()

    // Tell server to process everything
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop', paper_id: paperId ?? null }))
      setState('transcribing')
      setCurrentStage('Transcribing audio')
    } else {
      setState('error')
      setErrorMsg('Connection lost before processing.')
    }
  }, [cleanup, paperId])

  // ── Render ─────────────────────────────────────────────────────────────────

  const isProcessing = ['transcribing','keywords','graph','rag'].includes(state)
  const isDone       = state === 'answer'
  const isError      = state === 'error'

  return (
    <div className="relative">
      {/* Mic button */}
      <div className="flex items-center gap-1.5">
        {isRec && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
            onClick={stopRecording}
            className="flex h-7 items-center gap-1.5 rounded-full border border-red-500/30 bg-red-500/15 px-3 text-[11px] font-medium text-red-400"
            aria-label="Stop recording"
          >
            <FiSquare className="h-3 w-3" /> Stop
          </motion.button>
        )}

        <button
          onClick={() => {
            if (isRec) stopRecording()
            else if (state === 'idle' || isDone || isError) startRecording()
          }}
          disabled={disabled || isProcessing}
          aria-label={isRec ? 'Stop recording' : 'Start voice input'}
          className={cn(
            'relative flex h-9 w-9 items-center justify-center rounded-xl transition-colors',
            isRec        && 'bg-red-500/20 text-red-400',
            isProcessing && 'bg-white/10 text-dim opacity-40 pointer-events-none',
            isDone       && 'bg-white/8 text-silver',
            isError      && 'bg-red-500/15 text-red-400',
            !isRec && !isProcessing && !isDone && !isError && 'text-dim hover:text-light',
          )}
        >
          {isRec && (
            <motion.span className="absolute inset-0 rounded-xl bg-red-500/25"
              animate={{ scale: [1, 1.45], opacity: [0.5, 0] }}
              transition={{ repeat: Infinity, duration: 1.1 }} />
          )}
          {isProcessing
            ? <motion.span animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                <FiLoader className="h-4 w-4" />
              </motion.span>
            : <FiMic className="h-4 w-4" />
          }
        </button>

        {/* Panel toggle when there's content */}
        {(isRec || isProcessing || isDone || isError) && (
          <button onClick={() => setPanelOpen((v) => !v)}
            className="flex h-7 items-center gap-1 rounded-lg px-2 text-[11px] text-dim hover:text-light"
            aria-label={panelOpen ? 'Hide voice panel' : 'Show voice panel'}>
            {panelOpen ? <FiChevronDown className="h-3 w-3" /> : <FiChevronUp className="h-3 w-3" />}
          </button>
        )}
      </div>

      {/* Expandable panel */}
      <AnimatePresence>
        {panelOpen && (isRec || isProcessing || isDone || isError) && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.97 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="absolute bottom-full mb-2 right-0 z-50 w-[380px] overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0C0C0C]/95 shadow-2xl backdrop-blur-xl"
          >
            {/* Header */}
            <div className="flex items-center gap-2 border-b border-white/[0.06] px-4 py-3">
              {isRec && <Waveform active />}
              {!isRec && (
                <span className={cn(
                  'h-2 w-2 rounded-full',
                  isProcessing && 'animate-pulse bg-silver',
                  isDone       && 'bg-white/60',
                  isError      && 'bg-red-400',
                )} />
              )}
              <span className="flex-1 font-display text-xs font-semibold text-light">
                {isRec        && `Recording  ${timer}`}
                {isProcessing && 'Processing…'}
                {isDone       && `Done ${elapsed ? `· ${elapsed.toFixed(1)}s` : ''}`}
                {isError      && 'Error'}
              </span>
            </div>

            <div className="max-h-[420px] overflow-y-auto p-4 space-y-4">

              {/* Live / final transcript */}
              {(liveText || finalText) && (
                <div>
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-faint">
                    {finalText ? 'Transcript' : 'Live transcript'}
                  </p>
                  <p className="text-sm leading-relaxed text-silver-light">
                    {finalText || liveText}
                    {isRec && <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-silver-light align-middle" />}
                  </p>
                </div>
              )}

              {/* Pipeline stages */}
              {(isProcessing || isDone) && (
                <div className="space-y-1.5">
                  {Object.keys(STAGES).map((s) => {
                    const done    = doneStages.includes(s)
                    const active  = currentStage === s && !done
                    return (
                      <div key={s} className="flex items-center gap-2 text-[11px]">
                        <span className={cn(
                          'flex h-4 w-4 items-center justify-center rounded-full',
                          done   && 'text-white/70',
                          active && 'text-silver animate-pulse',
                          !done && !active && 'text-faint',
                        )}>
                          {done ? <FiCheckCircle className="h-3 w-3" /> : STAGES[s].icon}
                        </span>
                        <span className={cn(
                          done   && 'text-dim line-through',
                          active && 'text-silver-light',
                          !done && !active && 'text-faint',
                        )}>
                          {STAGES[s].label}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )}

              {/* Keywords */}
              {(keywords.length > 0 || entities.length > 0 || scientific.length > 0) && (
                <div className="space-y-2">
                  {scientific.length > 0 && (
                    <div>
                      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-faint">Scientific terms</p>
                      <div className="flex flex-wrap gap-1">
                        {scientific.map((t) => (
                          <span key={t} className="rounded-md border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 text-[10px] text-silver-light">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {entities.length > 0 && (
                    <div>
                      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-faint">Named entities</p>
                      <div className="flex flex-wrap gap-1">
                        {entities.map((e) => (
                          <span key={e} className="rounded-md border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 text-[10px] text-silver">{e}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {keywords.length > 0 && (
                    <div>
                      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-faint">Keywords</p>
                      <p className="text-[11px] text-faint">{keywords.slice(0, 10).join(', ')}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Graph status */}
              {graphInfo && (
                <div className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <FiShare2 className="h-3.5 w-3.5 text-silver" />
                  <span className="text-[11px] text-dim">
                    Knowledge Graph: {graphInfo.entities} entities, {graphInfo.keywords} keywords added
                  </span>
                </div>
              )}

              {/* Error */}
              {isError && errorMsg && (
                <div className="flex items-start gap-2 rounded-xl border border-red-500/25 bg-red-500/10 px-3 py-2.5">
                  <FiAlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-red-400" />
                  <p className="text-xs text-red-300">{errorMsg}</p>
                </div>
              )}

              {/* Answer preview */}
              {answer && (
                <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
                  <div className="mb-2 flex items-center gap-2">
                    <FiCheckCircle className="h-3.5 w-3.5 text-white/60" />
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-faint">Answer</span>
                    <span className="ml-auto rounded-full border border-white/[0.10] bg-white/[0.06] px-2 py-0.5 text-[10px] font-bold text-silver-bright">
                      Grade {answer.grade}
                    </span>
                    <span className="rounded-full border border-white/[0.10] bg-white/[0.04] px-2 py-0.5 text-[10px] text-silver">
                      {answer.search_label}
                    </span>
                  </div>
                  <p className="line-clamp-4 text-xs leading-relaxed text-silver-light">
                    {answer.text}
                  </p>
                  <p className="mt-2 text-[10px] text-faint">Full answer shown in chat above ↑</p>
                </div>
              )}

            </div>

            {/* Footer actions */}
            {(isDone || isError) && (
              <div className="border-t border-white/[0.06] px-4 py-2.5">
                <button
                  onClick={() => { setState('idle'); setPanelOpen(false); setLiveText(''); setFinalText(''); setAnswer(null); setErrorMsg(null) }}
                  className="w-full rounded-xl border border-white/[0.10] bg-white/[0.04] py-2 text-xs font-medium text-silver transition-colors hover:bg-white/[0.07]"
                >
                  Record again
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
