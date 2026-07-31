'use client'
import { forwardRef, type InputHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

interface Props extends InputHTMLAttributes<HTMLInputElement> { label?: string; error?: string }

export const Input = forwardRef<HTMLInputElement, Props>(function Input({ label, error, className, ...props }, ref) {
  return (
    <label className="flex flex-col gap-1.5">
      {label && <span className="text-xs font-medium text-dim">{label}</span>}
      <input ref={ref}
        className={cn('w-full rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-light outline-none transition-colors placeholder:text-faint focus:border-glow/50',
          error && 'border-red-500/50', className)}
        {...props} />
      {error && <span className="text-xs text-red-400">{error}</span>}
    </label>
  )
})
