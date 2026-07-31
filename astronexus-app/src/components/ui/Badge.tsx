import { type ReactNode } from 'react'
import { cn } from '@/lib/cn'
export function Badge({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-2 rounded-full border border-glow/25 bg-glow/10 px-3.5 py-1.5 text-xs font-medium tracking-wide text-glow-soft backdrop-blur-sm', className)}>
      {children}
    </span>
  )
}
