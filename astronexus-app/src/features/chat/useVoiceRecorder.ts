'use client'
/**
 * useVoiceRecorder
 *
 * Records microphone audio with MediaRecorder, then POSTs the blob to
 * the existing POST /voice/transcribe endpoint and returns the transcript.
 *
 * This is the simplest possible integration: no WebSocket, no PCM conversion,
 * no float32 streaming.  The browser produces a WebM/OGG blob; the server
 * writes it to disk, calls WhisperService().transcribe(), and returns the text.
 *
 * The backend /voice/transcribe endpoint already exists and is production-ready.
 * This hook only handles the recording + upload side.
 */

import { useRef, useState, useCallback } from 'react'
import { apiConfig } from '@/lib/config'

// ── Types ─────────────────────────────────────────────────────────────────────

export type RecorderState =
  | 'idle'
  | 'requesting'   // waiting for mic permission
  | 'recording'
  | 'uploading'    // blob sent, waiting for server
  | 'done'
  | 'error'

export interface UseVoiceRecorderReturn {
  state:       RecorderState
  transcript:  string
  elapsed:     number           // recording seconds (live)
  errorMsg:    string | null
  start:       () => Promise<void>
  stop:        () => void
  reset:       () => void
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useVoiceRecorder(
  onTranscript: (text: string) => void,
): UseVoiceRecorderReturn {
  const [state,      setState]     = useState<RecorderState>('idle')
  const [transcript, setTranscript]= useState('')
  const [elapsed,    setElapsed]   = useState(0)
  const [errorMsg,   setErrorMsg]  = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef        = useRef<Blob[]>([])
  const streamRef        = useRef<MediaStream | null>(null)
  const timerRef         = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Cleanup ────────────────────────────────────────────────────────────────

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }, [])

  // ── Start recording ────────────────────────────────────────────────────────

  const start = useCallback(async () => {
    setErrorMsg(null)
    setTranscript('')
    setElapsed(0)
    setState('requesting')

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
    } catch (e) {
      const denied = e instanceof DOMException && e.name === 'NotAllowedError'
      setErrorMsg(
        denied
          ? 'Microphone access denied. Allow it in your browser settings and try again.'
          : `Could not access microphone: ${e instanceof Error ? e.message : String(e)}`,
      )
      setState('error')
      return
    }

    // Pick the best supported MIME type
    const mimeType =
      MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')           ? 'audio/webm'
      : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')? 'audio/ogg;codecs=opus'
      : ''

    chunksRef.current = []
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    mediaRecorderRef.current = recorder

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    // onstop is triggered by stop() below — upload happens there
    recorder.onstop = async () => {
      cleanup()

      if (chunksRef.current.length === 0) {
        setErrorMsg('Recording was empty. Please try again.')
        setState('error')
        return
      }

      const mType = chunksRef.current[0].type || mimeType || 'audio/webm'
      const ext   = mType.includes('ogg') ? 'ogg' : 'webm'
      const blob  = new Blob(chunksRef.current, { type: mType })

      if (blob.size < 1000) {
        setErrorMsg('Recording too short. Please speak for at least one second.')
        setState('error')
        return
      }

      // ── POST to the existing /voice/transcribe endpoint ─────────────────────
      setState('uploading')
      const form = new FormData()
      form.append('audio', blob, `recording.${ext}`)

      // Attach JWT if present
      let headers: Record<string, string> = {}
      try {
        const raw = localStorage.getItem('anx-auth')
        const tok = raw ? JSON.parse(raw)?.state?.token : null
        if (tok) headers = { Authorization: `Bearer ${tok}` }
      } catch { /* no token */ }

      try {
        const controller = new AbortController()
        const timeout    = setTimeout(() => controller.abort(), 60_000) // 60s max

        const res = await fetch(`${apiConfig.baseUrl}/voice/transcribe`, {
          method:  'POST',
          headers,
          body:    form,
          signal:  controller.signal,
        })
        clearTimeout(timeout)

        if (!res.ok) {
          let detail = `Server error ${res.status}`
          try { detail = (await res.json()).detail ?? detail } catch { /**/ }
          throw new Error(detail)
        }

        const data = await res.json()

        if (!data.success) {
          throw new Error(data.error || 'Transcription failed — check server logs.')
        }

        if (!data.transcript?.trim()) {
          setErrorMsg('No speech detected. Please speak clearly and try again.')
          setState('error')
          return
        }

        setTranscript(data.transcript)
        onTranscript(data.transcript)   // → fills the chat input box
        setState('done')

      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') {
          setErrorMsg('Request timed out (60s). Try a shorter recording.')
        } else if (!navigator.onLine) {
          setErrorMsg('You appear to be offline. Check your connection and retry.')
        } else {
          setErrorMsg(
            e instanceof Error ? e.message : 'Transcription failed. Is the backend running?',
          )
        }
        setState('error')
      }
    }

    recorder.start(200)   // collect chunks every 200ms
    setState('recording')

    // Live timer
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000)
  }, [cleanup, onTranscript])

  // ── Stop recording ─────────────────────────────────────────────────────────

  const stop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()   // triggers recorder.onstop above
    }
  }, [])

  // ── Reset ──────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    stop()
    cleanup()
    setState('idle')
    setTranscript('')
    setElapsed(0)
    setErrorMsg(null)
    chunksRef.current = []
  }, [stop, cleanup])

  return { state, transcript, elapsed, errorMsg, start, stop, reset }
}
