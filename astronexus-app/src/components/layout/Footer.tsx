'use client'
import Link from 'next/link'
import { FiGithub } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { siteConfig } from '@/lib/config'

const COLS = [
  { title: 'Product', links: [['Dashboard', '/dashboard'], ['Chat', '/chat'], ['Knowledge Graph', '/knowledge-graph'], ['Vision AI', '/vision-ai']] },
  { title: 'Company', links: [['About', '/about'], ['Research', '/research'], ['Papers', '/papers']] },
]

export function Footer() {
  return (
    <footer className="relative z-10 border-t border-white/[0.06] bg-void/60 px-6 py-16 backdrop-blur-sm">
      <div className="mx-auto grid max-w-6xl gap-12 md:grid-cols-4">
        <div className="md:col-span-2">
          <Logo />
          <p className="mt-4 max-w-xs text-sm leading-relaxed text-dim">{siteConfig.description}</p>
          <a href="https://github.com" aria-label="GitHub"
            className="mt-6 inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 text-dim transition-colors hover:border-glow/40 hover:text-light">
            <FiGithub className="h-4 w-4" />
          </a>
        </div>
        {COLS.map((col) => (
          <div key={col.title}>
            <h4 className="mb-4 text-sm font-semibold text-light">{col.title}</h4>
            <ul className="space-y-2.5">
              {col.links.map(([label, href]) => (
                <li key={href}><Link href={href} className="text-sm text-dim transition-colors hover:text-glow-soft">{label}</Link></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mx-auto mt-12 max-w-6xl border-t border-white/[0.06] pt-8 text-center text-xs text-faint">
        © 2024 {siteConfig.name} · Built at NIT Kurukshetra · Research. Reason. Visualize. Discover.
      </div>
    </footer>
  )
}
