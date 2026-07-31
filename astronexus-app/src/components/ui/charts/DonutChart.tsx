'use client'
import { motion } from 'framer-motion'

export function DonutChart({ value, color = '#6CA2C1', size = 120, label }: { value: number; color?: string; size?: number; label?: string }) {
  const r = size / 2 - 10
  const circ = 2 * Math.PI * r
  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
        <motion.circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={circ} initial={{ strokeDashoffset: circ }}
          whileInView={{ strokeDashoffset: circ * (1 - value) }} viewport={{ once: true }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color}88)` }} />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-display text-xl font-bold text-light">{Math.round(value * 100)}%</span>
        {label && <span className="text-[10px] text-dim">{label}</span>}
      </div>
    </div>
  )
}
