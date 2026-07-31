'use client'
import { useEffect, useRef } from 'react'
import { useReducedMotion } from '@/hooks/useReducedMotion'

export function MouseFollower() {
  const reduced = useReducedMotion()
  const ref = useRef<HTMLDivElement>(null)
  const pos = useRef({ x: 0, y: 0 })
  const target = useRef({ x: 0, y: 0 })
  useEffect(() => {
    if (reduced || window.matchMedia('(pointer: coarse)').matches) return
    const move = (e: MouseEvent) => { target.current = { x: e.clientX, y: e.clientY } }
    window.addEventListener('mousemove', move, { passive: true })
    let raf = 0
    const loop = () => {
      pos.current.x += (target.current.x - pos.current.x) * 0.12
      pos.current.y += (target.current.y - pos.current.y) * 0.12
      if (ref.current) ref.current.style.transform = `translate(${pos.current.x - 140}px, ${pos.current.y - 140}px)`
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => { window.removeEventListener('mousemove', move); cancelAnimationFrame(raf) }
  }, [reduced])
  if (reduced) return null
  return <div ref={ref} aria-hidden className="pointer-events-none fixed left-0 top-0 z-20 h-[280px] w-[280px] rounded-full opacity-[0.06] blur-3xl"
    style={{ background: 'radial-gradient(circle, #6CA2C1, transparent 60%)' }} />
}
