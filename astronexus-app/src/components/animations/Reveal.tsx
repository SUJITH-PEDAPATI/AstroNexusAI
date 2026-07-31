'use client'
import { motion, type Variants } from 'framer-motion'
import { type ReactNode } from 'react'
import { fadeUp, blurReveal, scaleIn } from '@/lib/motion'
import { useReducedMotion } from '@/hooks/useReducedMotion'

const PRESETS: Record<string, Variants> = { fade: fadeUp, blur: blurReveal, scale: scaleIn }

export function Reveal({
  children, preset = 'fade', index = 0, className,
}: { children: ReactNode; preset?: 'fade' | 'blur' | 'scale'; index?: number; className?: string }) {
  const reduced = useReducedMotion()
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div className={className} variants={PRESETS[preset]} custom={index}
      initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-70px' }}>
      {children}
    </motion.div>
  )
}
