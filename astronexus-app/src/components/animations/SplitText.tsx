'use client'
import { motion } from 'framer-motion'
import { EASE } from '@/lib/motion'
import { useReducedMotion } from '@/hooks/useReducedMotion'
import { cn } from '@/lib/cn'

export function SplitText({
  text, className, delay = 0, by = 'word',
}: { text: string; className?: string; delay?: number; by?: 'char' | 'word' }) {
  const reduced = useReducedMotion()
  const units = by === 'char' ? text.split('') : text.split(' ')
  if (reduced) return <span className={className}>{text}</span>
  return (
    <span className={cn('inline-block', className)}>
      {units.map((u, i) => (
        <span key={i} className="inline-block overflow-hidden align-bottom">
          <motion.span className="inline-block" initial={{ y: '110%' }} whileInView={{ y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: delay + i * (by === 'char' ? 0.02 : 0.05), duration: 0.75, ease: EASE.out }}>
            {u}{by === 'word' && i < units.length - 1 ? '\u00A0' : ''}
          </motion.span>
        </span>
      ))}
    </span>
  )
}
