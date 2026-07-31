'use client'
import { useRef } from 'react'
import Link from 'next/link'
import { motion, useScroll, useTransform } from 'framer-motion'
import { FiArrowRight, FiVolume2, FiVolumeX } from 'react-icons/fi'
import { TbSatellite } from 'react-icons/tb'
import { SplitText } from '@/components/animations/SplitText'
import { Badge } from '@/components/ui/Badge'
import { EASE } from '@/lib/motion'
import { useReducedMotion } from '@/hooks/useReducedMotion'
import { useHeroSound } from './useHeroSound'

/**
 * Fullscreen cinematic hero built around the uploaded 4K video (with its
 * original soundtrack). Autoplays muted; an elegant control fades the audio in
 * and remembers the choice. On scroll the video scales + fades and the copy
 * lifts — an Apple-style handoff into the story.
 */
export function VideoHero() {
  const reduced = useReducedMotion()
  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const { soundEnabled, toggle } = useHeroSound(videoRef)

  const { scrollYProgress } = useScroll({ target: containerRef, offset: ['start start', 'end start'] })
  const scale = useTransform(scrollYProgress, [0, 1], [1, 1.12])
  const opacity = useTransform(scrollYProgress, [0, 0.85], [1, 0.15])
  const contentY = useTransform(scrollYProgress, [0, 1], [0, -110])
  const contentOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0])

  return (
    <section ref={containerRef} className="relative h-screen w-full overflow-hidden">
      <motion.div
        className="absolute inset-0"
        style={reduced ? undefined : { scale, opacity, willChange: 'transform', backfaceVisibility: 'hidden' }}
      >
        <video ref={videoRef} className="h-full w-full object-cover" autoPlay muted loop playsInline
          preload="auto" poster="/images/hero-poster.jpg" aria-hidden
          style={{ transform: 'translateZ(0)', backfaceVisibility: 'hidden' }}>
          {/* Highest-fidelity source first. Browsers use the first source they can
              play, so the ~2 Mbps H.264 master is chosen over the smaller VP9
              re-encode — every browser now gets the sharper file. */}
          <source src="/videos/hero.mp4" type="video/mp4" />
          <source src="/videos/hero.webm" type="video/webm" />
        </video>
      </motion.div>

      {/* Cinematic grading (inherits the video's palette) */}
      <div className="absolute inset-0 bg-gradient-to-b from-void/45 via-void/10 to-void" />
      <div className="absolute inset-0 bg-gradient-to-r from-void/80 via-transparent to-transparent" />
      <div className="pointer-events-none absolute inset-0" style={{ boxShadow: 'inset 0 0 180px 40px rgba(7,8,10,0.72)' }} />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-void to-transparent" />

      {/* Content */}
      <motion.div className="relative z-10 flex h-full items-center px-6" style={reduced ? undefined : { y: contentY, opacity: contentOpacity }}>
        <div className="mx-auto w-full max-w-6xl">
          <div className="max-w-2xl">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.8, ease: EASE.out }}>
              <Badge><TbSatellite className="h-3.5 w-3.5" /> Space Intelligence Platform</Badge>
            </motion.div>
            <h1 className="mt-6 font-display text-5xl font-bold leading-[1.03] tracking-tight text-light md:text-7xl">
              <SplitText text="Intelligence for" delay={0.4} />
              <br />
              <span className="bg-gradient-to-r from-glow via-blue-bright to-gold bg-clip-text text-transparent">
                <SplitText text="the Universe" delay={0.62} />
              </span>
            </h1>
            <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.05, duration: 0.8, ease: EASE.out }}
              className="mt-7 max-w-xl text-lg leading-relaxed text-dim">
              The AI operating system for space science. Research papers, knowledge graphs,
              satellite vision, and multi-agent reasoning — unified in one cinematic platform.
            </motion.p>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.25, duration: 0.8, ease: EASE.out }}
              className="mt-9 flex flex-wrap items-center gap-4">
              <Link href="/dashboard" className="group inline-flex items-center gap-2 rounded-full bg-blue px-7 py-3.5 text-sm font-medium text-white shadow-[0_0_28px_rgba(59,130,246,0.4)] transition-colors hover:bg-blue-bright">
                Launch platform <FiArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </Link>
              <Link href="/about" className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.04] px-7 py-3.5 text-sm font-medium text-light backdrop-blur-md transition-colors hover:border-glow/50">
                Watch the story
              </Link>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Sound toggle */}
      <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 }}
        onClick={toggle}
        className="absolute bottom-8 right-6 z-20 flex items-center gap-2 rounded-full border border-white/12 bg-void/50 px-4 py-2.5 text-xs text-light backdrop-blur-md transition-colors hover:border-glow/50"
        aria-label={soundEnabled ? 'Mute soundtrack' : 'Play soundtrack'}>
        {soundEnabled ? <FiVolume2 className="h-4 w-4 text-glow-soft" /> : <FiVolumeX className="h-4 w-4" />}
        {soundEnabled ? 'Sound on' : 'Sound off'}
      </motion.button>

      {/* Scroll cue */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 }}
        className="absolute bottom-8 left-1/2 z-10 flex -translate-x-1/2 flex-col items-center gap-2">
        <span className="text-[10px] uppercase tracking-[0.3em] text-dim/70">Scroll</span>
        <motion.span animate={{ y: [0, 8, 0] }} transition={{ repeat: Infinity, duration: 1.6 }}
          className="h-8 w-px bg-gradient-to-b from-glow/70 to-transparent" />
      </motion.div>
    </section>
  )
}
