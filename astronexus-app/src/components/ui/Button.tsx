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
  primary: 'bg-blue text-white hover:bg-blue-bright shadow-[0_0_24px_rgba(59,130,246,0.35)]',
  secondary: 'bg-white/[0.04] border border-white/12 text-light hover:border-glow/50 backdrop-blur-md',
  ghost: 'text-dim hover:text-light hover:bg-white/5',
  danger: 'bg-red-500/15 border border-red-500/30 text-red-300 hover:bg-red-500/25',
}
const SIZES: Record<Size, string> = {
  sm: 'px-3.5 py-2 text-xs rounded-lg',
  md: 'px-5 py-2.5 text-sm rounded-xl',
  lg: 'px-7 py-3.5 text-sm rounded-full',
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
