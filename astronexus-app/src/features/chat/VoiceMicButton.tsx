'use client'
/**
 * VoiceMicButton
 *
 * Self-contained microphone button for the chat composer.
 * Uses useVoiceRecorder → POST /voice/transcribe (existing endpoint).
 *
 * States:
 *   idle        → mic icon, click to start
 *   requesting  → spinner, waiting for permission dialog
 *   recording   → pulsing red button + timer + waveform bars + Stop chip
 *   uploading   → spinner, "Processing…"
 *   done        → transcript written to chat input (parent handles display)
 *   error       → red alert above the button
 *
 * The transcript is delivered via onTranscript(text) and populates the
 * chat input box. The user can then edit it and send normally.
 */

import { useEffect, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiMic, FiSquare, FiLoader, FiAlertCircle } from 'react-icons/fi'
import { useVoiceRecorder } from './useVoiceRecorder'
import { cn } from '@/lib/cn'

// ── Waveform bars ─────────────────────────────────────────────────────────────

const Waveform = memo(function Waveform() {
  return (
    <div className="flex items-center gap-0.5" aria-hidden>
      {Array.from({ length: 12 }).map((_, i) => (
        <motion.span
          key={i}
          className="w-0.5 rounded-full bg-red-400"
          animate={{ height: ['3px', `${5 + Math.random() * 14}px`, '3px'] }}
          transition={{
            repeat:   Infinity,
            duration: 0.45 + Math.random() * 0.35,
            delay:    i * 0.05,
            ease:     'easeInOut',
          }}
        />
      ))}
    </div>
  )
})

// ── Timer formatter ────────────────────────────────────────────────────────────

function fmtTime(s: number): string {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  onTranscript: (text: string) => void
  disabled?:    boolean
}

// ── Component ─────────────────────────────────────────────────────────────────

export function VoiceMicButton({ onTranscript, disabled = false }: Props) {
  const { state, elapsed, errorMsg, start, stop, reset } =
    useVoiceRecorder(onTranscript)

  const isRecording  = state === 'recording'
  const isProcessing = state === 'uploading' || state === 'requesting'
  const isError      = state === 'error'
  const isDone       = state === 'done'

  // Auto-clear error after 6 s
  useEffect(() => {
    if (!isError) return
    const t = setTimeout(reset, 6000)
    return () => clearTimeout(t)
  }, [isError, reset])

  // Reset done state after 2 s (input is already filled)
  useEffect(() => {
    if (!isDone) return
    const t = setTimeout(reset, 2000)
    return () => clearTimeout(t)
  }, [isDone, reset])

  return (
    <div className="relative flex items-center gap-1.5">
      {/* Error tooltip — appears above the button */}
      <AnimatePresence>
        {isError && errorMsg && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.2 }}
            className="absolute bottom-full mb-2 right-0 z-50 flex w-[280px] items-start gap-2 rounded-xl border border-red-500/25 bg-[#0C0C0C]/95 px-3 py-2.5 shadow-lg backdrop-blur-xl"
            role="alert"
          >
            <FiAlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-red-400" />
            <p className="text-xs leading-snug text-red-300">{errorMsg}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Recording status chip (timer + waveform + stop button) */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2 rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1.5"
          >
            <Waveform />
            <span className="font-mono text-[11px] tabular-nums text-red-400">
              {fmtTime(elapsed)}
            </span>
            <button
              onClick={stop}
              className="flex items-center gap-1 rounded-full bg-red-500/20 px-2 py-0.5 text-[11px] font-medium text-red-400 transition-colors hover:bg-red-500/30"
              aria-label="Stop recording"
            >
              <FiSquare className="h-2.5 w-2.5" /> Stop
            </button>
          </motion.div>
        )}

        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-1.5 text-[11px] text-dim"
          >
            <motion.span
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
            >
              <FiLoader className="h-3 w-3" />
            </motion.span>
            {state === 'requesting' ? 'Waiting for microphone…' : 'Processing…'}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main mic button */}
      <button
        onClick={() => {
          if (isRecording) stop()
          else if (state === 'idle' || isDone || isError) start()
        }}
        disabled={disabled || isProcessing}
        aria-label={
          isRecording  ? 'Stop recording'
          : isProcessing ? 'Processing audio…'
          : 'Record voice input'
        }
        className={cn(
          'relative flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl transition-colors duration-200',
          isRecording  && 'bg-red-500/20 text-red-400',
          isProcessing && 'cursor-not-allowed opacity-40',
          isDone       && 'text-silver',
          isError      && 'text-red-400',
          !isRecording && !isProcessing && !isDone && !isError
            && 'text-dim hover:bg-white/5 hover:text-light',
          disabled && 'cursor-not-allowed opacity-40',
        )}
      >
        {/* Pulse ring while recording */}
        {isRecording && (
          <motion.span
            className="absolute inset-0 rounded-xl bg-red-500/20"
            animate={{ scale: [1, 1.5], opacity: [0.4, 0] }}
            transition={{ repeat: Infinity, duration: 1.1 }}
          />
        )}

        {isProcessing
          ? (
            <motion.span
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
            >
              <FiLoader className="h-4 w-4" />
            </motion.span>
          )
          : <FiMic className="h-4 w-4" />
        }
      </button>
    </div>
  )
}
