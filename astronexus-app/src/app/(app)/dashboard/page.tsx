'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { FiArrowUpRight, FiArrowRight, FiBookmark } from 'react-icons/fi'
import { Card } from '@/components/ui/Card'
import { AreaChart } from '@/components/ui/charts/AreaChart'
import { BarChart } from '@/components/ui/charts/BarChart'
import { DonutChart } from '@/components/ui/charts/DonutChart'
import { useAuth } from '@/hooks/useAuth'
import { STAT_CARDS, USAGE_TREND, QUERY_BARS, RECENT_ACTIVITY, BOOKMARKS } from '@/features/dashboard/data'
import { stagger, fadeUp } from '@/lib/motion'

export default function DashboardPage() {
  const { user } = useAuth()
  return (
    <div className="mx-auto max-w-7xl p-6 md:p-8">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold text-light md:text-3xl">
          Welcome back, {user?.name?.split(' ')[0] ?? 'Researcher'}
        </h1>
        <p className="mt-1 text-sm text-dim">Here&apos;s what&apos;s happening across your research workspace.</p>
      </div>

      {/* Stat cards */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STAT_CARDS.map((s) => {
          const Icon = s.icon
          return (
            <motion.div key={s.label} variants={fadeUp}>
              <Card hover className="p-5">
                <div className="flex items-start justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl" style={{ background: `${s.accent}1a`, border: `1px solid ${s.accent}33` }}>
                    <Icon className="h-5 w-5" style={{ color: s.accent }} />
                  </div>
                  <span className="flex items-center gap-0.5 text-xs font-medium text-emerald-400"><FiArrowUpRight className="h-3 w-3" />{s.delta}%</span>
                </div>
                <div className="mt-4 font-display text-3xl font-bold text-light">{s.value}</div>
                <div className="mt-1 text-sm text-dim">{s.label}</div>
              </Card>
            </motion.div>
          )
        })}
      </motion.div>

      {/* Charts row */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="font-display text-sm font-semibold text-light">Usage trend</h3>
              <p className="text-xs text-dim">Queries per week</p>
            </div>
            <span className="text-xs text-glow-soft">+34% this month</span>
          </div>
          <AreaChart data={USAGE_TREND} height={200} />
        </Card>
        <Card className="flex flex-col items-center justify-center p-6">
          <h3 className="mb-4 self-start font-display text-sm font-semibold text-light">Answer reliability</h3>
          <DonutChart value={0.86} label="avg grade" size={150} />
        </Card>
      </div>

      {/* Bottom row */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="p-6">
          <h3 className="mb-4 font-display text-sm font-semibold text-light">Weekly queries</h3>
          <BarChart data={QUERY_BARS} height={180} />
        </Card>
        <Card className="p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-display text-sm font-semibold text-light">Recent activity</h3>
            <Link href="/chat/history" className="text-xs text-glow-soft hover:underline">View all</Link>
          </div>
          <div className="space-y-3">
            {RECENT_ACTIVITY.map((a, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ background: a.accent }} />
                <div className="min-w-0 flex-1">
                  <span className="text-xs text-dim">{a.action} </span>
                  <span className="truncate text-xs text-light">{a.target}</span>
                </div>
                <span className="flex-shrink-0 text-[10px] text-faint">{a.time}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card className="p-6">
          <h3 className="mb-4 font-display text-sm font-semibold text-light">Bookmarks</h3>
          <div className="space-y-2">
            {BOOKMARKS.map((b, i) => (
              <Link key={i} href="/chat" className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-xs text-dim transition-colors hover:border-glow/30 hover:text-light">
                <FiBookmark className="h-3.5 w-3.5 flex-shrink-0 text-glow-soft" />
                <span className="truncate">{b}</span>
              </Link>
            ))}
          </div>
          <Link href="/chat" className="mt-4 flex items-center justify-center gap-1.5 rounded-xl bg-blue/15 py-2.5 text-xs font-medium text-glow-soft transition-colors hover:bg-blue/25">
            New conversation <FiArrowRight className="h-3.5 w-3.5" />
          </Link>
        </Card>
      </div>
    </div>
  )
}
