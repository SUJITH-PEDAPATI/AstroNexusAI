'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { FiLogOut, FiChevronLeft } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Avatar } from '@/components/ui/Avatar'
import { APP_NAV, APP_NAV_FOOTER } from '@/lib/navigation'
import { useUIStore } from '@/store/ui.store'
import { useAuth } from '@/hooks/useAuth'
import { cn } from '@/lib/cn'

export function AppSidebar() {
  const pathname = usePathname()
  const { sidebarOpen, toggleSidebar } = useUIStore()
  const { user, signOut } = useAuth()

  const isActive = (href: string) =>
    pathname === href || (href !== '/chat' && pathname.startsWith(href + '/')) ||
    (href === '/chat' && pathname === '/chat')

  return (
    <motion.aside animate={{ width: sidebarOpen ? 264 : 76 }} transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="anx-sidebar relative z-30 hidden h-screen flex-shrink-0 flex-col border-r border-white/[0.06] md:flex">
      <div className={cn('flex h-16 items-center border-b border-white/[0.06] px-4', sidebarOpen ? 'justify-between' : 'justify-center')}>
        {sidebarOpen ? <Logo /> : <Logo compact />}
        {sidebarOpen && (
          <button onClick={toggleSidebar} className="rounded-lg p-1.5 text-dim hover:bg-white/5 hover:text-light" aria-label="Collapse sidebar">
            <FiChevronLeft className="h-4 w-4" />
          </button>
        )}
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto p-3">
        {APP_NAV.map((item) => {
          const Icon = item.icon
          const active = isActive(item.href)
          return (
            <Link key={item.href} href={item.href}
              className={cn('group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors',
                active ? 'anx-nav-active text-light' : 'text-dim hover:bg-white/5 hover:text-light',
                !sidebarOpen && 'justify-center')}>
              {active && <motion.span layoutId="sidebar-active" className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r bg-white/60" />}
              <Icon className={cn('h-[18px] w-[18px] flex-shrink-0', active && 'text-glow-soft')} />
              <AnimatePresence>
                {sidebarOpen && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="whitespace-nowrap">{item.label}</motion.span>}
              </AnimatePresence>
            </Link>
          )
        })}
      </nav>

      <div className="space-y-1 border-t border-white/[0.06] p-3">
        {APP_NAV_FOOTER.map((item) => {
          const Icon = item.icon
          const active = isActive(item.href)
          return (
            <Link key={item.href} href={item.href}
              className={cn('flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors',
                active ? 'anx-nav-active text-light' : 'text-dim hover:bg-white/5 hover:text-light', !sidebarOpen && 'justify-center')}>
              <Icon className="h-[18px] w-[18px] flex-shrink-0" />
              {sidebarOpen && <span>{item.label}</span>}
            </Link>
          )
        })}
        {user && (
          <div className={cn('mt-2 flex items-center gap-3 rounded-xl bg-white/[0.03] p-2', !sidebarOpen && 'justify-center')}>
            <Avatar name={user.name} color={user.avatarColor} size={32} />
            {sidebarOpen && (
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-medium text-light">{user.name}</div>
                <div className="truncate text-[10px] text-faint">{user.email}</div>
              </div>
            )}
            {sidebarOpen && (
              <button onClick={signOut} className="rounded-lg p-1.5 text-dim hover:bg-white/5 hover:text-red-400" aria-label="Sign out">
                <FiLogOut className="h-4 w-4" />
              </button>
            )}
          </div>
        )}
      </div>
    </motion.aside>
  )
}
