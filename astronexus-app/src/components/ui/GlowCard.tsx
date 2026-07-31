'use client'
import { useRef, type ReactNode } from 'react'
import { motion, useMotionValue, useTransform } from 'framer-motion'
import { cn } from '@/lib/cn'
import { useReducedMotion } from '@/hooks/useReducedMotion'

export function GlowCard({ children, className, accent = '#6CA2C1' }: { children: ReactNode; className?: string; accent?: string }) {
  const reduced = useReducedMotion()
  const ref = useRef<HTMLDivElement>(null)
  const mx = useMotionValue(0)
  const my = useMotionValue(0)
  const rx = useTransform(my, [-120, 120], [5, -5])
  const ry = useTransform(mx, [-120, 120], [-5, 5])

  const onMove = (e: React.MouseEvent) => {
    if (reduced) return
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    mx.set(e.clientX - r.left - r.width / 2)
    my.set(e.clientY - r.top - r.height / 2)
  }
  return (
    <motion.div ref={ref} onMouseMove={onMove} onMouseLeave={() => { mx.set(0); my.set(0) }}
      style={reduced ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: 900 }}
      className={cn('group relative overflow-hidden rounded-2xl border border-white/[0.07] bg-white/[0.025] p-6 transition-colors duration-300 hover:border-white/[0.14]', className)}>
      <div className="pointer-events-none absolute -inset-px rounded-2xl opacity-0 transition-opacity duration-500 group-hover:opacity-100"
        style={{ background: `radial-gradient(240px circle at 50% 0px, ${accent}1f, transparent 70%)` }} />
      <div className="relative z-10">{children}</div>
    </motion.div>
  )
}
