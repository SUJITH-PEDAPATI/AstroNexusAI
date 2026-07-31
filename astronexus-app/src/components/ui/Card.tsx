'use client'
import { type ReactNode } from 'react'
import { cn } from '@/lib/cn'

export function Card({ children, className, hover = false }: { children: ReactNode; className?: string; hover?: boolean }) {
  return (
    <div className={cn(
      'rounded-2xl border border-white/[0.07] bg-white/[0.025] backdrop-blur-sm',
      hover && 'transition-colors duration-300 hover:border-white/[0.14]',
      className,
    )}>
      {children}
    </div>
  )
}
