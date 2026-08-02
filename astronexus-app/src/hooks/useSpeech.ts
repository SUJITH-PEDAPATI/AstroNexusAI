'use client'
import { useRef, useState, useEffect, useCallback } from 'react'

export interface UseSpeechReturn {
  speak:      (text: string, id: string) => void
  stop:       () => void
  isSpeaking: (id: string) => boolean
  supported:  boolean
}

/**
 * Browser SpeechSynthesis hook.
 * One utterance at a time — calling speak() while another is active stops it first.
 * Cleans up automatically on unmount.
 */
export function useSpeech(): UseSpeechReturn {
  const supported     = typeof window !== 'undefined' && 'speechSynthesis' in window
  const [activeId, setActiveId] = useState<string | null>(null)
  const utteranceRef  = useRef<SpeechSynthesisUtterance | null>(null)

  // Cancel on unmount
  useEffect(() => () => { if (supported) window.speechSynthesis.cancel() }, [supported])

  const stop = useCallback(() => {
    if (!supported) return
    window.speechSynthesis.cancel()
    setActiveId(null)
    utteranceRef.current = null
  }, [supported])

  const speak = useCallback((text: string, id: string) => {
    if (!supported || !text.trim()) return

    // Stop anything already playing
    window.speechSynthesis.cancel()

    // If clicking the same message that's playing, just stop
    if (activeId === id) { setActiveId(null); return }

    const utterance   = new SpeechSynthesisUtterance(text.replace(/\*+|#{1,6}|`{1,3}|\[|\]/g, ''))
    utterance.lang    = 'en-US'
    utterance.rate    = 1.0
    utterance.pitch   = 1.0
    utterance.volume  = 1.0

    // Pick an English voice if the browser provides one
    const voices = window.speechSynthesis.getVoices()
    const enVoice = voices.find((v) => v.lang.startsWith('en'))
    if (enVoice) utterance.voice = enVoice

    utterance.onend   = () => { setActiveId(null); utteranceRef.current = null }
    utterance.onerror = () => { setActiveId(null); utteranceRef.current = null }

    utteranceRef.current = utterance
    setActiveId(id)
    window.speechSynthesis.speak(utterance)
  }, [supported, activeId])

  const isSpeaking = useCallback((id: string) => activeId === id, [activeId])

  return { speak, stop, isSpeaking, supported }
}
