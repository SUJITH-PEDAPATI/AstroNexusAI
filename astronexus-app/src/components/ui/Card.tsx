'use client'
import { type ReactNode } from 'react'
import { cn } from '@/lib/cn'

export function Card({ children, className, hover = false }: { children: ReactNode; className?: string; hover?: boolean }) {
  return (
    <div className={cn(
      'anx-card',
      hover && 'cursor-default',
      className,
    )}>
      {children}
    </div>
  )
}
