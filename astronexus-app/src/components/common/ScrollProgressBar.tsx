'use client'
import { useState, useEffect } from 'react'
export function ScrollProgressBar() {
  const [p, setP] = useState(0)
  useEffect(() => {
    const h = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      setP(max > 0 ? window.scrollY / max : 0)
    }
    h(); window.addEventListener('scroll', h, { passive: true })
    return () => window.removeEventListener('scroll', h)
  }, [])
  return (
    <div className="fixed left-0 top-0 z-50 h-[2px] w-full bg-transparent">
      <div className="h-full bg-gradient-to-r from-glow via-blue-bright to-glow-soft"
        style={{ width: `${p * 100}%`, boxShadow: '0 0 10px rgba(108,162,193,0.7)' }} />
    </div>
  )
}
