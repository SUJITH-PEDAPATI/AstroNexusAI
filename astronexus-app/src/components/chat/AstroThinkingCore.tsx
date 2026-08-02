'use client'
/**
 * AstroThinkingCore — AstroNexus AI's signature thinking indicator.
 *
 * A futuristic energy core with an orbital ring and orbiting particles.
 * Replaces the traditional loading spinner in the chat during inference.
 *
 *   idle:      soft breathing pulse, particles drift
 *   thinking:  ring rotates, particles orbit, core pulses brighter
 *   completed: ring slows, glow softens, then fades out
 *
 * 100% SVG + Framer Motion. No GIFs, no Lottie, no external assets.
 * Memoised. Respects prefers-reduced-motion. Pauses when hidden.
 */

import { memo, useEffect, useState } from 'react'
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion'

// ── Colors (CSS variable–friendly) ────────────────────────────────────────────

// Monochrome palette — white / silver, matching the graphite design system
const C = {
  primary:   '#B4B4B4',   // silver core
  secondary: '#D8D8D8',   // bright silver ring
  accent:    '#F2F2F2',   // near-white highlights
  blue:      '#8A8A8A',   // muted steel (secondary ring)
} as const

// ── Types ─────────────────────────────────────────────────────────────────────

interface AstroThinkingCoreProps {
  /** Whether the thinking animation is active. */
  active: boolean
  /** Current pipeline stage text, e.g. "Searching Vector Database". */
  currentStage?: string
  /** 0–1 fractional progress (not currently used, reserved for SSE). */
  progress?: number
}

// ── Stage simulation (replaces real SSE until backend streams stages) ─────────

const STAGES = [
  '✓ Understanding question',
  '✓ Detecting intent',
  '✓ Routing query',
  '⏳ Searching Research Papers…',
  '⏳ Querying Knowledge Graph…',
  '⏳ Searching Trusted Web Sources…',
  '✓ Ranking results',
  '⏳ Generating response…',
  '✓ Generating citations',
]

function useStageProgress(active: boolean, externalStage?: string) {
  const [stages, setStages] = useState<string[]>([])
  const [idx, setIdx]       = useState(0)

  useEffect(() => {
    if (!active) { setStages([]); setIdx(0); return }
    if (externalStage) return // skip simulation when backend provides real stages

    let i = 0
    setStages([STAGES[0]])
    setIdx(0)
    const timer = setInterval(() => {
      i++
      if (i < STAGES.length) {
        setStages((prev) => [...prev, STAGES[i]])
        setIdx(i)
      } else {
        clearInterval(timer)
      }
    }, 1400)
    return () => clearInterval(timer)
  }, [active, externalStage])

  // If backend provides a real stage, just show that one
  if (externalStage) return { stages: [externalStage], currentIdx: 0 }
  return { stages, currentIdx: idx }
}

// ── Particle config ───────────────────────────────────────────────────────────

const PARTICLES = [
  { angle: 0,   r: 18, size: 2.2, speed: 6,  color: C.accent   },
  { angle: 120, r: 19, size: 1.6, speed: 8,  color: C.secondary },
  { angle: 240, r: 17, size: 1.8, speed: 10, color: C.blue      },
]

// ── Component ─────────────────────────────────────────────────────────────────

export const AstroThinkingCore = memo(function AstroThinkingCore({
  active,
  currentStage,
}: AstroThinkingCoreProps) {
  const prefersReduced = useReducedMotion()
  const { stages, currentIdx } = useStageProgress(active, currentStage)

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center gap-4 py-4"
        >
          {/* ── SVG mascot ──────────────────────────────────────────── */}
          <div className="relative flex h-12 w-12 items-center justify-center">
            <svg
              viewBox="0 0 48 48"
              width={48}
              height={48}
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="overflow-visible"
              aria-label="AstroNexus is thinking"
              role="img"
            >
              <defs>
                {/* Core radial glow */}
                <radialGradient id="anx-core-glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor={C.primary}   stopOpacity="0.9" />
                  <stop offset="55%"  stopColor={C.secondary} stopOpacity="0.5" />
                  <stop offset="100%" stopColor={C.primary}   stopOpacity="0" />
                </radialGradient>

                {/* Outer atmospheric haze */}
                <radialGradient id="anx-haze" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor={C.accent} stopOpacity="0.12" />
                  <stop offset="100%" stopColor={C.accent} stopOpacity="0" />
                </radialGradient>

                {/* Inner constellation pattern */}
                <radialGradient id="anx-inner" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor={C.accent}  stopOpacity="0.7" />
                  <stop offset="100%" stopColor={C.primary} stopOpacity="0.2" />
                </radialGradient>
              </defs>

              {/* Atmospheric haze (outermost glow) */}
              <motion.circle
                cx={24} cy={24} r={22}
                fill="url(#anx-haze)"
                animate={prefersReduced ? {} : { opacity: [0.4, 0.7, 0.4] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
              />

              {/* Orbital ring */}
              <motion.ellipse
                cx={24} cy={24} rx={18} ry={6}
                stroke={C.secondary}
                strokeWidth={0.6}
                strokeOpacity={0.5}
                fill="none"
                animate={prefersReduced ? {} : { rotate: 360 }}
                transition={prefersReduced ? {} : { repeat: Infinity, duration: 6, ease: 'linear' }}
                style={{ transformOrigin: '24px 24px' }}
              />

              {/* Second orbital ring (tilted, more subtle) */}
              <motion.ellipse
                cx={24} cy={24} rx={16} ry={5}
                stroke={C.blue}
                strokeWidth={0.4}
                strokeOpacity={0.25}
                fill="none"
                transform="rotate(60 24 24)"
                animate={prefersReduced ? {} : { rotate: -360 }}
                transition={prefersReduced ? {} : { repeat: Infinity, duration: 10, ease: 'linear' }}
                style={{ transformOrigin: '24px 24px' }}
              />

              {/* Core body */}
              <motion.circle
                cx={24} cy={24} r={7}
                fill="url(#anx-core-glow)"
                animate={prefersReduced
                  ? { opacity: [0.8, 1, 0.8] }
                  : { r: [6.5, 7.5, 6.5], opacity: [0.85, 1, 0.85] }
                }
                transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
              />

              {/* Inner core (bright centre) */}
              <motion.circle
                cx={24} cy={24} r={3.5}
                fill="url(#anx-inner)"
                animate={prefersReduced ? {} : { r: [3, 4, 3], opacity: [0.7, 1, 0.7] }}
                transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut', delay: 0.2 }}
              />

              {/* Inner constellation geometry — six-pointed star */}
              <motion.path
                d="M24 20.5 L25.3 23 L28 23.5 L26 25.5 L26.5 28 L24 26.8 L21.5 28 L22 25.5 L20 23.5 L22.7 23 Z"
                fill={C.accent}
                fillOpacity={0.5}
                animate={prefersReduced ? {} : { fillOpacity: [0.3, 0.6, 0.3], scale: [0.95, 1.05, 0.95] }}
                transition={{ repeat: Infinity, duration: 2.4, ease: 'easeInOut' }}
                style={{ transformOrigin: '24px 24px' }}
              />

              {/* Orbiting particles */}
              {!prefersReduced && PARTICLES.map((p, i) => (
                <motion.circle
                  key={i}
                  r={p.size}
                  fill={p.color}
                  fillOpacity={0.85}
                  animate={{
                    cx: [
                      24 + p.r * Math.cos((p.angle * Math.PI) / 180),
                      24 + p.r * Math.cos(((p.angle + 120) * Math.PI) / 180),
                      24 + p.r * Math.cos(((p.angle + 240) * Math.PI) / 180),
                      24 + p.r * Math.cos((p.angle * Math.PI) / 180),
                    ],
                    cy: [
                      24 + (p.r * 0.3) * Math.sin((p.angle * Math.PI) / 180),
                      24 + (p.r * 0.3) * Math.sin(((p.angle + 120) * Math.PI) / 180),
                      24 + (p.r * 0.3) * Math.sin(((p.angle + 240) * Math.PI) / 180),
                      24 + (p.r * 0.3) * Math.sin((p.angle * Math.PI) / 180),
                    ],
                    opacity: [0.9, 0.5, 0.9],
                  }}
                  transition={{ repeat: Infinity, duration: p.speed, ease: 'linear' }}
                />
              ))}

              {/* Occasional sparkle (thinking only) */}
              {!prefersReduced && (
                <motion.circle
                  cx={32} cy={16} r={1}
                  fill={C.accent}
                  animate={{ opacity: [0, 0, 1, 0], scale: [0.5, 0.5, 1.5, 0.5] }}
                  transition={{ repeat: Infinity, duration: 3.5, ease: 'easeInOut', delay: 1.2 }}
                  style={{ transformOrigin: '32px 16px' }}
                />
              )}
            </svg>
          </div>

          {/* ── Label ───────────────────────────────────────────────── */}
          <p className="font-display text-xs font-medium tracking-wide"
            style={{ color: C.secondary }}>
            AstroNexus is thinking…
          </p>

          {/* ── Pipeline stage log ──────────────────────────────────── */}
          {stages.length > 0 && (
            <div className="flex flex-col gap-1">
              {stages.map((s, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                  className="font-mono text-[11px]"
                  style={{ color: i === currentIdx && s.startsWith('⏳') ? C.secondary : '#64748B' }}
                >
                  {s}
                </motion.span>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
})
