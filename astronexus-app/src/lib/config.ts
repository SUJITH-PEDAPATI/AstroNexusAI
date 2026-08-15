export const siteConfig = {
  name: 'AstroNexusAI',
  tagline: 'Intelligence for the Universe',
  description:
    'An AI-powered space research platform unifying RAG, knowledge graphs, satellite intelligence, and multi-agent reasoning.',
  url: 'https://astronexus.ai',
} as const

export const apiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws',
  endpoints: {
    login: '/auth/login',
    register: '/auth/register',
    chat: '/chat',
    conversations: '/conversations',
    ingest: '/ingest',
    papers: '/papers',
    graph: '/graph',
    graphTopics: '/graph/topics',
    vision: '/vision',
    dashboard: '/dashboard/stats',
    clearPapers: '/papers',
  },
} as const
