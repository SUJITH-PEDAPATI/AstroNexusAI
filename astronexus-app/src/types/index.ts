import type { IconType } from 'react-icons'

export type DeviceTier = 'low' | 'mid' | 'high'

export interface User {
  id: string
  name: string
  email: string
  avatarColor: string
  avatarUrl?: string
  provider: 'email' | 'google' | 'github'
  createdAt: string
}

export interface Citation {
  section: string
  page: number
  score: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  grade?: string
  createdAt: number
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  folder?: string
  pinned: boolean
  favorite: boolean
  createdAt: number
  updatedAt: number
}

export interface Paper {
  id: string
  title: string
  authors: string[]
  chunks: number
  status: 'processing' | 'ready' | 'failed'
  uploadedAt: number
  keywords: string[]
}

export interface GraphNodeT {
  id: string
  label: string
  type: 'Paper' | 'Author' | 'Keyword' | 'Domain' | 'Satellite' | 'Institution'
  val?: number
}
export interface GraphLinkT { source: string; target: string; label: string }

export interface NavItem {
  label: string
  href: string
  icon: IconType
}

export interface StatCard {
  label: string
  value: string
  delta: number
  icon: IconType
  accent: string
}
