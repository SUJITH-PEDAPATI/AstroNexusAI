'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { FiMenu, FiBell, FiSearch } from 'react-icons/fi'
import { useUIStore } from '@/store/ui.store'
import { Fragment } from 'react'

function titleCase(s: string) {
  return s.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function AppTopbar() {
  const pathname = usePathname()
  const toggleSidebar = useUIStore((s) => s.toggleSidebar)
  const segments = pathname.split('/').filter(Boolean)

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-white/[0.06] bg-space/70 px-4 backdrop-blur-xl md:px-6">
      <div className="flex items-center gap-3">
        <button onClick={toggleSidebar} className="rounded-lg p-2 text-dim hover:bg-white/5 hover:text-light" aria-label="Toggle sidebar">
          <FiMenu className="h-5 w-5" />
        </button>
        <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-sm">
          <Link href="/dashboard" className="text-dim hover:text-light">Home</Link>
          {segments.map((seg, i) => {
            const href = '/' + segments.slice(0, i + 1).join('/')
            const last = i === segments.length - 1
            return (
              <Fragment key={href}>
                <span className="text-faint">/</span>
                {last ? <span className="text-light">{titleCase(seg)}</span>
                  : <Link href={href} className="text-dim hover:text-light">{titleCase(seg)}</Link>}
              </Fragment>
            )
          })}
        </nav>
      </div>
      <div className="flex items-center gap-2">
        <button className="rounded-lg p-2 text-dim hover:bg-white/5 hover:text-light" aria-label="Search"><FiSearch className="h-[18px] w-[18px]" /></button>
        <button className="relative rounded-lg p-2 text-dim hover:bg-white/5 hover:text-light" aria-label="Notifications">
          <FiBell className="h-[18px] w-[18px]" />
          <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-glow" />
        </button>
      </div>
    </header>
  )
}
