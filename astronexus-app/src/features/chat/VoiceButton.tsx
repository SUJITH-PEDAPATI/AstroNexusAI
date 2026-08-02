'use client'
/**
 * VoiceButton — microphone input for the chat composer.
 *
 * What it does:
 *   1. Records audio via MediaRecorder (WebM/Opus, supported everywhere)
 *   2. POSTs the blob to POST /voice/chat on the existing backend
 *   3. Calls onTranscript(text) so the parent can populate the textarea
 *   4. Plays the TTS audio_url returned by the backend
 *   5. Calls onAnswer(text, grade) so the parent can render the reply bubble
 *
 * What it does NOT touch:
 *   - The existing send() / chat store / message rendering — untouched
 *   - Any backend inference logic — untouched
 */

import { useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiMic, FiMicOff, FiVolume2 } from 'react-icons/fi'
import { apiConfig } from '@/lib/config'
import { cn } from '@/lib/cn'

type VoiceState = 'idle' | 'recording' | 'processing' | 'playing'

interface Props {
  /** Called with the Whisper transcript so the textarea shows what was heard. */
  onTranscript: (text: string) => void
  /** Called with the AI answer + grade so the chat store can add the bubble. */
  onAnswer: (answer: string, grade: string) => void
  /** Disable while the text send is already in-flight. */
  disabled?: boolean
}

export function VoiceButton({ onTranscript, onAnswer, disabled = false }: Props) {
  const [state, setState] = useState<VoiceState>('idle')
  const [error, setError]   = useState<string | null>(null)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunks        = useRef<Blob[]>([])
  const audioEl       = useRef<HTMLAudioElement | null>(null)

  // ── helpers ──────────────────────────────────────────────────────────────

  const stopAudio = () => {
    if (audioEl.current) {
      audioEl.current.pause()
      audioEl.current.src = ''
    }
  }

  // ── submit to backend ────────────────────────────────────────────────────

  const submitAudio = useCallback(async () => {
    if (!chunks.current.length) { setState('idle'); return }

    const mimeType = chunks.current[0].type || 'audio/webm'
    const ext      = mimeType.includes('mp4') ? 'mp4' : 'webm'
    const blob     = new Blob(chunks.current, { type: mimeType })

    const form = new FormData()
    form.append('audio', blob, `voice_query.${ext}`)

    try {
      const res = await fetch(`${apiConfig.baseUrl}/voice/chat`, {
        method: 'POST',
        body:   form,
        // No Content-Type header — browser sets the correct multipart boundary
        headers: (() => {
          const h: Record<string, string> = {}
          try {
            const raw = localStorage.getItem('anx-auth')
            const tok = raw ? JSON.parse(raw)?.state?.token : null
            if (tok) h['Authorization'] = `Bearer ${tok}`
          } catch { /* no token */ }
          return h
        })(),
      })

      if (!res.ok) {
        let detail = `Error ${res.status}`
        try { detail = (await res.json()).detail ?? detail } catch { /**/ }
        throw new Error(detail)
      }

      const data = await res.json() as {
        transcript:  string
        answer:      string
        audio_url:   string
        confidence:  string
        is_reliable: boolean
      }

      // 1. Populate textarea with what was heard
      onTranscript(data.transcript)

      // 2. Send answer + grade to parent to render as a chat bubble
      const gradeMap: Record<string, string> = { HIGH: 'A', MEDIUM: 'B', LOW: 'C' }
      onAnswer(data.answer, gradeMap[data.confidence] ?? 'B')

      // 3. Play TTS audio if backend returned a URL
      if (data.audio_url) {
        setState('playing')
        const audio = new Audio(`${apiConfig.baseUrl}${data.audio_url}`)
        audioEl.current = audio
        audio.onended = () => setState('idle')
        audio.onerror = () => setState('idle')
        audio.play().catch(() => setState('idle'))
        return
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice request failed.')
    }

    setState('idle')
  }, [onTranscript, onAnswer])

  // ── record ───────────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setError(null)
    stopAudio()

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone access denied.')
      return
    }

    chunks.current = []

    // Prefer WebM/Opus; fall back to whatever the browser supports
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : ''

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    mediaRecorder.current = recorder

    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.current.push(e.data) }
    recorder.onstop = () => { stream.getTracks().forEach((t) => t.stop()); submitAudio() }

    recorder.start(250)   // collect chunks every 250 ms
    setState('recording')
  }, [submitAudio])

  const stopRecording = () => {
    if (mediaRecorder.current?.state === 'recording') {
      mediaRecorder.current.stop()
      setState('processing')
    }
  }



  // ── render ───────────────────────────────────────────────────────────────

  const isRecording  = state === 'recording'
  const isProcessing = state === 'processing'
  const isPlaying    = state === 'playing'

  return (
    <div className="relative flex items-center">
      {/* Error tooltip */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-lg border border-red-500/30 bg-void/90 px-3 py-1.5 text-[10px] text-red-400 backdrop-blur"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stop button shown while recording */}
      {isRecording && (
        <motion.button
          initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
          onClick={stopRecording}
          className="mr-1 flex h-7 items-center gap-1.5 rounded-full bg-red-500/15 px-3 text-[11px] font-medium text-red-400 border border-red-500/30"
          aria-label="Stop recording"
        >
          <FiMicOff className="h-3 w-3" /> Stop
        </motion.button>
      )}

      {/* Mic / status button */}
      <button
        onClick={() => {
          if (state === 'recording') stopRecording()
          else if (state === 'playing') { stopAudio(); setState('idle') }
          else if (state === 'idle') startRecording()
        }}
        disabled={disabled || isProcessing}
        aria-label={
          isRecording  ? 'Stop recording' :
          isProcessing ? 'Processing…'    :
          isPlaying    ? 'Stop playback'  :
          'Start voice input'
        }
        className={cn(
          'relative flex h-9 w-9 items-center justify-center rounded-xl transition-colors',
          isRecording  && 'bg-red-500/20 text-red-400',
          isProcessing && 'bg-white/10 text-dim',
          isPlaying    && 'bg-glow/20 text-glow-soft',
          !isRecording && !isProcessing && !isPlaying && 'text-dim hover:text-light',
          (disabled || isProcessing) && 'pointer-events-none opacity-40',
        )}
      >
        {/* Pulse ring while recording */}
        {isRecording && (
          <motion.span
            className="absolute inset-0 rounded-xl bg-red-500/20"
            animate={{ scale: [1, 1.4], opacity: [0.5, 0] }}
            transition={{ repeat: Infinity, duration: 1.1 }}
          />
        )}
        {isPlaying
          ? <FiVolume2 className="h-4 w-4" />
          : <FiMic    className="h-4 w-4" />
        }
      </button>
    </div>
  )
}
