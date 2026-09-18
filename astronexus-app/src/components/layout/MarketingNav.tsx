'use client'
import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { FiArrowUpRight } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { MARKETING_NAV } from '@/lib/navigation'
import { EASE } from '@/lib/motion'
import { cn } from '@/lib/cn'

/** Floating, glass navbar for public pages. */
export function MarketingNav() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', h, { passive: true })
    return () => window.removeEventListener('scroll', h)
  }, [])
  return (
    <motion.header initial={{ y: -70, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: EASE.out }}
      className="fixed inset-x-0 top-4 z-40 px-4">
      <nav className={cn('mx-auto flex max-w-5xl items-center justify-between rounded-2xl border px-5 py-3 transition-all duration-500',
        scrolled ? 'border-white/10 bg-void/70 backdrop-blur-2xl' : 'border-white/[0.04] bg-void/30 backdrop-blur-md')}>
        <Logo />
        <div className="hidden items-center gap-1 md:flex">
          {MARKETING_NAV.map((l) => (
            <Link key={l.href} href={l.href}
              className="rounded-lg px-4 py-2 text-sm text-dim transition-colors hover:bg-white/5 hover:text-light">{l.label}</Link>
          ))}
        </div>
        <Link href='/dashboard'
          className="flex items-center gap-1.5 rounded-full bg-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-bright">
          {'Open App'} <FiArrowUpRight className="h-3.5 w-3.5" />
        </Link>
      </nav>
    </motion.header>
  )
}
