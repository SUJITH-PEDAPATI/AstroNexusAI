'use client'
import { motion } from 'framer-motion'

export function BarChart({ data, color = '#3B82F6', height = 160 }: { data: { label: string; value: number }[]; color?: string; height?: number }) {
  const max = Math.max(...data.map((d) => d.value), 1)
  return (
    <div className="flex items-end justify-between gap-2" style={{ height }}>
      {data.map((d, i) => (
        <div key={d.label} className="flex flex-1 flex-col items-center gap-2">
          <div className="flex w-full flex-1 items-end">
            <motion.div className="w-full rounded-t-md"
              style={{ background: `linear-gradient(to top, ${color}, ${color}77)` }}
              initial={{ height: 0 }} whileInView={{ height: `${(d.value / max) * 100}%` }}
              viewport={{ once: true }} transition={{ delay: i * 0.06, duration: 0.7, ease: [0.16, 1, 0.3, 1] }} />
          </div>
          <span className="text-[10px] text-faint">{d.label}</span>
        </div>
      ))}
    </div>
  )
}
