'use client'
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'lg'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  children: ReactNode
}

const VARIANTS: Record<Variant, string> = {
  primary:   'anx-button',
  secondary: 'bg-white/[0.04] border border-white/[0.10] text-light hover:bg-white/[0.07] hover:border-white/20 backdrop-blur-md',
  ghost:     'text-dim hover:text-light hover:bg-white/5',
  danger:    'bg-red-500/12 border border-red-500/25 text-red-300 hover:bg-red-500/20',
}
const SIZES: Record<Size, string> = {
  sm: 'px-3.5 py-2   text-xs rounded-[12px]',
  md: 'px-5   py-2.5 text-sm rounded-[16px]',
  lg: 'px-7   py-3.5 text-sm rounded-[16px]',
}

export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { variant = 'primary', size = 'md', className, children, ...props }, ref,
) {
  return (
    <motion.button ref={ref} whileTap={{ scale: 0.97 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      className={cn('inline-flex items-center justify-center gap-2 font-medium transition-colors duration-200 disabled:opacity-40 disabled:pointer-events-none',
        VARIANTS[variant], SIZES[size], className)}
      {...(props as any)}>
      {children}
    </motion.button>
  )
})
