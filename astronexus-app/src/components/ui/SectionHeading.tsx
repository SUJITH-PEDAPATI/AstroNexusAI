import { type ReactNode } from 'react'
import { Badge } from './Badge'
import { Reveal } from '@/components/animations/Reveal'
import { SplitText } from '@/components/animations/SplitText'
import { cn } from '@/lib/cn'

export function SectionHeading({
  badge, title, highlight, subtitle, align = 'center', className,
}: { badge?: string; title: string; highlight?: string; subtitle?: ReactNode; align?: 'left' | 'center'; className?: string }) {
  return (
    <div className={cn('flex flex-col gap-5', align === 'center' ? 'items-center text-center' : 'items-start', className)}>
      {badge && <Reveal><Badge>{badge}</Badge></Reveal>}
      <h2 className="max-w-3xl font-display text-4xl font-bold leading-[1.08] tracking-tight text-light md:text-5xl">
        <SplitText text={title} />
        {highlight && <> <span className="bg-gradient-to-r from-glow via-blue-bright to-glow-soft bg-clip-text text-transparent"><SplitText text={highlight} delay={0.12} /></span></>}
      </h2>
      {subtitle && <Reveal index={2}><p className={cn('max-w-2xl text-lg leading-relaxed text-dim', align === 'center' && 'mx-auto')}>{subtitle}</p></Reveal>}
    </div>
  )
}
