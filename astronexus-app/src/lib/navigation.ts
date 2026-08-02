import {
  FiGrid, FiMessageSquare, FiClock, FiSearch, FiShare2, FiEye, FiFileText, FiUser, FiSettings, FiMic,
} from 'react-icons/fi'
import type { NavItem } from '@/types'

/** Sidebar navigation for the authenticated app shell. */
export const APP_NAV: NavItem[] = [
  { label: 'Dashboard', href: '/dashboard', icon: FiGrid },
  { label: 'Chat', href: '/chat', icon: FiMessageSquare },
  { label: 'History', href: '/chat/history', icon: FiClock },
  { label: 'Research', href: '/research', icon: FiSearch },
  { label: 'Papers', href: '/papers', icon: FiFileText },
  { label: 'Knowledge Graph', href: '/knowledge-graph', icon: FiShare2 },
  { label: 'Vision AI', href: '/vision-ai', icon: FiEye },
]

export const APP_NAV_FOOTER: NavItem[] = [
  { label: 'Voice Test 🧪', href: '/voice-test', icon: FiMic },

  { label: 'Profile', href: '/profile', icon: FiUser },
  { label: 'Settings', href: '/settings', icon: FiSettings },
]

/** Top navbar for the public marketing pages. */
export const MARKETING_NAV = [
  { label: 'Home', href: '/' },
  { label: 'About', href: '/about' },
  { label: 'Dashboard', href: '/dashboard' },
]
