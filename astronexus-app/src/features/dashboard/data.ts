import { FiMessageSquare, FiFileText, FiZap, FiTrendingUp } from 'react-icons/fi'
import type { StatCard } from '@/types'

// ── Static fallbacks — used when the backend is offline ───────────────────────
// The dashboard page replaces these with live data when the API responds.

export const FALLBACK_STAT_CARDS: StatCard[] = [
  { label: 'Conversations', value: '—', delta: 0, icon: FiMessageSquare, accent: '#6CA2C1' },
  { label: 'Papers Ingested', value: '—', delta: 0, icon: FiFileText, accent: '#3B82F6' },
  { label: 'Queries Run', value: '—', delta: 0, icon: FiZap, accent: '#7DD3FC' },
  { label: 'Avg Reliability', value: '—', delta: 0, icon: FiTrendingUp, accent: '#D4B483' },
]

export const FALLBACK_USAGE_TREND = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
export const FALLBACK_QUERY_BARS = [
  { label: 'Mon', value: 0 }, { label: 'Tue', value: 0 }, { label: 'Wed', value: 0 },
  { label: 'Thu', value: 0 }, { label: 'Fri', value: 0 }, { label: 'Sat', value: 0 }, { label: 'Sun', value: 0 },
]
export const FALLBACK_RECENT_ACTIVITY: { action: string; target: string; time: string; accent: string }[] = []
export const FALLBACK_BOOKMARKS: string[] = []

// ── Backend response shape from GET /dashboard/stats ─────────────────────────
export interface DashboardStats {
  conversations: number
  papers: number
  queries: number
  avg_reliability: number
  usage_trend: number[]      // 12 weekly numbers
  query_bars: number[]       // 7 daily numbers (Mon–Sun)
  recent_activity: { action: string; target: string; time: string }[]
  bookmarks: string[]
}
