'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { TbTelescope } from 'react-icons/tb'
import { cn } from '@/lib/cn'

export function Logo({ href = '/', compact = false, className }: { href?: string; compact?: boolean; className?: string }) {
  return (
    <Link href={href} className={cn('group flex items-center gap-2.5', className)} aria-label="AstroNexusAI home">
      <motion.span whileHover={{ rotate: 12 }} transition={{ type: 'spring', stiffness: 300 }}
        className="flex h-9 w-9 items-center justify-center rounded-xl border border-glow/40 bg-glow/15">
        <TbTelescope className="h-5 w-5 text-glow-soft" />
      </motion.span>
      {!compact && (
        <span className="font-display text-lg font-bold tracking-tight text-light">
          Astro<span className="text-glow-soft">Nexus</span><span className="ml-0.5 text-xs text-glow">AI</span>
        </span>
      )}
    </Link>
  )
}
