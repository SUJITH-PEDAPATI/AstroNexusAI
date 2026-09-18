'use client'
import { useRef, useCallback, useEffect } from 'react'
import { useUIStore } from '@/store/ui.store'

/**
 * Manages the hero video soundtrack per the SRS audio spec:
 * - starts muted (autoplay always allowed)
 * - unmutes on first user interaction if the user opts in
 * - remembers the preference (persisted in the UI store)
 * - fades volume in/out smoothly instead of hard cutting
 */
export function useHeroSound(videoRef: React.RefObject<HTMLVideoElement>) {
  const { soundEnabled, setSound } = useUIStore()
  const fadeRaf = useRef(0)

  const fadeTo = useCallback((target: number) => {
    const v = videoRef.current
    if (!v) return
    cancelAnimationFrame(fadeRaf.current)
    const start = v.volume
    const t0 = performance.now()
    const dur = 600
    const step = (now: number) => {
      const p = Math.min((now - t0) / dur, 1)
      v.volume = start + (target - start) * p
      if (p < 1) fadeRaf.current = requestAnimationFrame(step)
    }
    fadeRaf.current = requestAnimationFrame(step)
  }, [videoRef])

  const enable = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = false
    v.volume = 0
    v.play().catch(() => {})
    fadeTo(1)
    setSound(true)
  }, [videoRef, fadeTo, setSound])

  const disable = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    fadeTo(0)
    setTimeout(() => { if (v) v.muted = true }, 600)
    setSound(false)
  }, [videoRef, fadeTo, setSound])

  const toggle = useCallback(() => {
    if (soundEnabled) disable()
    else enable()
  }, [soundEnabled, enable, disable])

  // Restore preference on mount (if user previously enabled sound, unmute on
  // first interaction to satisfy autoplay policies).
  useEffect(() => {
    if (!soundEnabled) return
    const onFirst = () => { enable(); window.removeEventListener('pointerdown', onFirst) }
    window.addEventListener('pointerdown', onFirst, { once: true })
    return () => window.removeEventListener('pointerdown', onFirst)
  }, [soundEnabled, enable])

  return { soundEnabled, toggle }
}
