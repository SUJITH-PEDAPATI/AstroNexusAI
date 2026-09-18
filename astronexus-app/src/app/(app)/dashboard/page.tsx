'use client'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { FiArrowUpRight, FiArrowRight, FiBookmark } from 'react-icons/fi'
import { useQuery } from '@tanstack/react-query'
import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { AreaChart } from '@/components/ui/charts/AreaChart'
import { BarChart } from '@/components/ui/charts/BarChart'
import { DonutChart } from '@/components/ui/charts/DonutChart'
import { http } from '@/services/http'
import { apiConfig } from '@/lib/config'
import { stagger, fadeUp } from '@/lib/motion'
import { useAuthStore } from '@/store/auth.store'
import {
  FALLBACK_STAT_CARDS, FALLBACK_USAGE_TREND, FALLBACK_QUERY_BARS,
  FALLBACK_RECENT_ACTIVITY, FALLBACK_BOOKMARKS,
  type DashboardStats,
} from '@/features/dashboard/data'
import { FiMessageSquare, FiFileText, FiZap, FiTrendingUp } from 'react-icons/fi'

const ICON_MAP = [FiMessageSquare, FiFileText, FiZap, FiTrendingUp]
const ACCENT_MAP = ['#8A8A8A', '#6E6E6E', '#B4B4B4', '#A8A49E']

export default function DashboardPage() {
  const { user } = useAuthStore()
  const { data, isLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const res = await http<DashboardStats>(apiConfig.endpoints.dashboard)
      if (!res) throw new Error('offline')
      return res
    },
    // Don't throw to the error boundary — we fall back gracefully
    retry: 1,
    staleTime: 60_000,
  })

  // Build stat cards from live data or fallbacks
  const statCards = data
    ? [
        { label: 'Conversations', value: String(data.conversations), delta: 0, icon: FiMessageSquare, accent: '#8A8A8A' },
        { label: 'Papers Ingested', value: String(data.papers), delta: 0, icon: FiFileText, accent: '#6E6E6E' },
        { label: 'Queries Run', value: String(data.queries), delta: 0, icon: FiZap, accent: '#B4B4B4' },
        { label: 'Avg Reliability', value: data.avg_reliability?.toFixed(2) ?? '—', delta: 0, icon: FiTrendingUp, accent: '#A8A49E' },
      ]
    : FALLBACK_STAT_CARDS

  const usageTrend = data?.usage_trend ?? FALLBACK_USAGE_TREND
  const queryBars = data?.query_bars
    ? data.query_bars.map((v, i) => ({ label: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i] ?? String(i), value: v }))
    : FALLBACK_QUERY_BARS
  const recentActivity = data?.recent_activity ?? FALLBACK_RECENT_ACTIVITY
  const bookmarks = data?.bookmarks ?? FALLBACK_BOOKMARKS
  const reliability = data?.avg_reliability ?? 0

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
        {statCards.map((s, idx) => {
          const Icon = s.icon ?? ICON_MAP[idx]
          return (
            <motion.div key={s.label} variants={fadeUp}>
              <Card hover className="p-5">
                <div className="flex items-start justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl"
                    style={{ background: `${s.accent ?? ACCENT_MAP[idx]}1a`, border: `1px solid ${s.accent ?? ACCENT_MAP[idx]}33` }}>
                    <Icon className="h-5 w-5" style={{ color: s.accent ?? ACCENT_MAP[idx] }} />
                  </div>
                  {s.delta > 0 && (
                    <span className="flex items-center gap-0.5 text-xs font-medium text-silver-light">
                      <FiArrowUpRight className="h-3 w-3" />{s.delta}%
                    </span>
                  )}
                </div>
                {isLoading
                  ? <Skeleton className="mt-4 h-8 w-20" />
                  : <div className="mt-4 font-display text-3xl font-bold text-light">{s.value}</div>}
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
          </div>
          {isLoading ? <Skeleton className="h-[200px] w-full" /> : <AreaChart data={usageTrend} height={200} />}
        </Card>
        <Card className="flex flex-col items-center justify-center p-6">
          <h3 className="mb-4 self-start font-display text-sm font-semibold text-light">Answer reliability</h3>
          {isLoading ? <Skeleton className="h-[150px] w-[150px] rounded-full" /> : <DonutChart value={reliability} label="avg grade" size={150} />}
        </Card>
      </div>

      {/* Bottom row */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="p-6">
          <h3 className="mb-4 font-display text-sm font-semibold text-light">Weekly queries</h3>
          {isLoading ? <Skeleton className="h-[180px] w-full" /> : <BarChart data={queryBars} height={180} />}
        </Card>
        <Card className="p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-display text-sm font-semibold text-light">Recent activity</h3>
            <Link href="/chat/history" className="text-xs text-glow-soft hover:underline">View all</Link>
          </div>
          <div className="space-y-3">
            {isLoading && <Skeleton className="h-24 w-full" />}
            {!isLoading && recentActivity.length === 0 && (
              <p className="text-xs text-faint">No activity yet.</p>
            )}
            {recentActivity.map((a, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="h-2 w-2 flex-shrink-0 rounded-full bg-glow" />
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
            {isLoading && <Skeleton className="h-24 w-full" />}
            {!isLoading && bookmarks.length === 0 && (
              <p className="text-xs text-faint">No bookmarks yet.</p>
            )}
            {bookmarks.map((b, i) => (
              <Link key={i} href="/chat"
                className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-xs text-dim transition-colors hover:border-glow/30 hover:text-light">
                <FiBookmark className="h-3.5 w-3.5 flex-shrink-0 text-glow-soft" />
                <span className="truncate">{b}</span>
              </Link>
            ))}
          </div>
          <Link href="/chat"
            className="mt-4 flex items-center justify-center gap-1.5 rounded-xl bg-blue/15 py-2.5 text-xs font-medium text-glow-soft transition-colors hover:bg-blue/25">
            New conversation <FiArrowRight className="h-3.5 w-3.5" />
          </Link>
        </Card>
      </div>
    </div>
  )
}
