import { FiMessageSquare, FiFileText, FiZap, FiTrendingUp } from 'react-icons/fi'
import type { StatCard } from '@/types'

export const STAT_CARDS: StatCard[] = [
  { label: 'Conversations', value: '128', delta: 12, icon: FiMessageSquare, accent: '#6CA2C1' },
  { label: 'Papers Ingested', value: '34', delta: 8, icon: FiFileText, accent: '#3B82F6' },
  { label: 'Queries Run', value: '1,204', delta: 23, icon: FiZap, accent: '#7DD3FC' },
  { label: 'Avg Reliability', value: '0.86', delta: 4, icon: FiTrendingUp, accent: '#D4B483' },
]

export const USAGE_TREND = [12, 18, 15, 24, 22, 31, 28, 36, 33, 42, 39, 48]
export const QUERY_BARS = [
  { label: 'Mon', value: 42 }, { label: 'Tue', value: 55 }, { label: 'Wed', value: 38 },
  { label: 'Thu', value: 68 }, { label: 'Fri', value: 72 }, { label: 'Sat', value: 30 }, { label: 'Sun', value: 25 },
]
export const RECENT_ACTIVITY = [
  { action: 'Queried', target: 'Origins Space Telescope', time: '2m ago', accent: '#6CA2C1' },
  { action: 'Ingested', target: 'exoplanet_atmospheres.pdf', time: '1h ago', accent: '#3B82F6' },
  { action: 'Exported', target: 'Chat: IR spectroscopy', time: '3h ago', accent: '#D4B483' },
  { action: 'Expanded', target: 'Knowledge graph: SIMBAD', time: '5h ago', accent: '#7DD3FC' },
]
export const BOOKMARKS = [
  'What is the OST far-infrared range?',
  'How does SAR detect flooding?',
  'Biosignature detection methods',
]
