'use client'
import dynamic from 'next/dynamic'
import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  FiRefreshCw, FiAlertCircle, FiInfo, FiSearch,
  FiChevronDown, FiX, FiDatabase, FiGitBranch, FiMaximize2, FiMinimize2,
} from 'react-icons/fi'
import { Spinner } from '@/components/ui/Spinner'
import { http, ApiError } from '@/services/http'
import { apiConfig } from '@/lib/config'

const GraphScene = dynamic(
  () => import('@/features/knowledge-graph/GraphScene').then(m => m.GraphScene),
  { ssr: false, loading: () => <div className="flex h-full items-center justify-center"><Spinner size={28} /></div> },
)

/* ── Types ──────────────────────────────────────────────────────────────── */
interface GNode  { id: string; label: string; type: string; val: number }
interface GLink  { source: string; target: string; label: string }
interface GraphData { nodes: GNode[]; links: GLink[]; meta: Record<string, unknown> }
interface PaperItem { id: string; title: string }
interface TopicItem { topic_id: string; label: string; query_count: number; keyword_count: number }

/* ── Colours ────────────────────────────────────────────────────────────── */
const TYPE_COLOR: Record<string, string> = {
  Paper: '#60A5FA', Topic: '#A78BFA', Author: '#34D399',
  Keyword: '#FBBF24', Entity: '#F472B6', Domain: '#2DD4BF',
  Model: '#818CF8', Dataset: '#FB923C', Task: '#38BDF8',
  VoiceInput: '#E879F9', Query: '#94A3B8',
}

/* ── Prebuilt topic categories from node types ──────────────────────────── */
const PREBUILT_TOPICS = [
  { id: '__all_papers',  label: 'All Papers',  icon: '📄', type: 'Paper'   },
  { id: '__all_models',  label: 'Models',      icon: '🧠', type: 'Model'   },
  { id: '__all_datasets',label: 'Datasets',    icon: '📊', type: 'Dataset' },
  { id: '__all_tasks',   label: 'Tasks',       icon: '🎯', type: 'Task'    },
  { id: '__all_domains', label: 'Domains',     icon: '🌌', type: 'Domain'  },
  { id: '__all_authors', label: 'Authors',     icon: '👤', type: 'Author'  },
  { id: '__all_entities',label: 'Entities',    icon: '🔬', type: 'Entity'  },
  { id: '__all_keywords',label: 'Keywords',    icon: '🏷', type: 'Keyword' },
]

type FilterMode = 'overview' | 'paper' | 'topic' | 'category'

/* ── Searchable dropdown ────────────────────────────────────────────────── */
function SearchDropdown<T extends { id: string; label: string }>({
  items, value, onChange, placeholder, renderItem,
}: {
  items: T[]
  value: string
  onChange: (v: string) => void
  placeholder: string
  renderItem?: (item: T) => React.ReactNode
}) {
  const [open, setOpen]     = useState(false)
  const [search, setSearch] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = items.filter(i => i.label.toLowerCase().includes(search.toLowerCase()))
  const selected = items.find(i => i.id === value)

  return (
    <div ref={ref} className="relative w-full max-w-md">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-2 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-left text-sm transition-colors hover:border-white/[0.15] hover:bg-white/[0.05]"
      >
        <span className={selected ? 'text-light' : 'text-faint'}>
          {selected?.label ?? placeholder}
        </span>
        <div className="flex items-center gap-1.5">
          {value && (
            <span onClick={e => { e.stopPropagation(); onChange(''); setSearch('') }}
              className="rounded p-0.5 text-faint hover:text-light">
              <FiX className="h-3 w-3" />
            </span>
          )}
          <FiChevronDown className={`h-3.5 w-3.5 text-faint transition-transform ${open ? 'rotate-180' : ''}`} />
        </div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 mt-1.5 w-full rounded-xl border border-white/[0.10] bg-[#0A0A14]/98 shadow-2xl backdrop-blur-xl"
          >
            {/* Search */}
            <div className="flex items-center gap-2 border-b border-white/[0.06] px-3 py-2">
              <FiSearch className="h-3.5 w-3.5 text-faint" />
              <input
                type="text" value={search} onChange={e => setSearch(e.target.value)}
                placeholder="Search…" autoFocus
                className="flex-1 bg-transparent text-sm text-light outline-none placeholder:text-faint"
              />
            </div>
            {/* Items */}
            <div className="max-h-56 overflow-y-auto py-1 scrollbar-thin">
              {filtered.length === 0 && (
                <p className="px-4 py-3 text-xs text-faint">No results</p>
              )}
              {filtered.map(item => (
                <button key={item.id}
                  onClick={() => { onChange(item.id); setOpen(false); setSearch('') }}
                  className={`flex w-full items-center gap-2.5 px-4 py-2 text-left text-sm transition-colors hover:bg-white/[0.06] ${
                    item.id === value ? 'bg-white/[0.04] text-light' : 'text-silver-light'
                  }`}>
                  {renderItem ? renderItem(item) : item.label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ── Error helper ───────────────────────────────────────────────────────── */
function describeError(err: unknown): { title: string; hint: string } {
  if (err instanceof ApiError) {
    if (err.status === 404) return {
      title: 'Graph endpoint not found (404)',
      hint: 'Copy server_final.py → backend/api/server.py and restart uvicorn.',
    }
    if (err.status >= 500) return { title: `Backend error (${err.status})`, hint: err.message }
    return { title: `API error ${err.status}`, hint: err.message }
  }
  if (err instanceof TypeError) return {
    title: 'Cannot reach backend',
    hint: `Check that uvicorn is running on ${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}`,
  }
  return { title: 'Unknown error', hint: String(err) }
}

/* ── Page ───────────────────────────────────────────────────────────────── */
export default function KnowledgeGraphPage() {
  const [mode,       setMode]       = useState<FilterMode>('overview')
  const [paperId,    setPaperId]    = useState('')
  const [topicId,    setTopicId]    = useState('')
  const [categoryId, setCategoryId] = useState('')

  // ── Fullscreen ──────────────────────────────────────────────────────────
  const [isFullScreen, setIsFullScreen] = useState(false)
  const graphWrapRef = useRef<HTMLDivElement>(null)

  const toggleFullScreen = useCallback(() => {
    const el = graphWrapRef.current
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().catch(() => {
        // Fallback: CSS-only fullscreen
        setIsFullScreen(true)
      })
    } else {
      document.exitFullscreen?.()
    }
  }, [])

  // Sync state when browser fires fullscreenchange (Esc key, etc.)
  useEffect(() => {
    const handler = () => setIsFullScreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  // Escape key fallback for CSS-only fullscreen
  useEffect(() => {
    if (!isFullScreen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !document.fullscreenElement) {
        setIsFullScreen(false)
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [isFullScreen])

  // Build query string
  const qs = mode === 'paper' && paperId
    ? `?paper_id=${encodeURIComponent(paperId)}`
    : mode === 'topic' && topicId
    ? `?topic_id=${encodeURIComponent(topicId)}`
    : ''
  const graphUrl = `${apiConfig.endpoints.graph}${qs}`

  // Fetch graph data
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery<GraphData, Error>({
    queryKey: ['graph', mode, paperId, topicId, categoryId],
    queryFn: async () => {
      const r = await http<GraphData>(graphUrl)
      if (r === null) throw new TypeError('Network error')
      return r
    },
    staleTime: 30_000, retry: 1,
  })

  // Paper list
  const { data: papers } = useQuery<PaperItem[]>({
    queryKey: ['papers'],
    queryFn: () => http<PaperItem[]>(apiConfig.endpoints.papers).then(r => r ?? []),
    staleTime: 60_000,
  })

  // Topic list
  const { data: topics } = useQuery<TopicItem[]>({
    queryKey: ['graph-topics'],
    queryFn: () => http<TopicItem[]>(apiConfig.endpoints.graphTopics).then(r => r ?? []),
    staleTime: 30_000,
  })

  // Filter by category (client-side filter on the overview data)
  const rawNodes = data?.nodes ?? []
  const rawLinks = data?.links ?? []

  const { filteredNodes, filteredLinks } = useMemo(() => {
    if (mode !== 'category' || !categoryId) {
      return { filteredNodes: rawNodes, filteredLinks: rawLinks }
    }
    const cat = PREBUILT_TOPICS.find(c => c.id === categoryId)
    if (!cat) return { filteredNodes: rawNodes, filteredLinks: rawLinks }

    // Keep all nodes of this type + their direct parents
    const typeNodes = new Set(rawNodes.filter(n => n.type === cat.type).map(n => n.id))
    const connected = new Set(typeNodes)
    for (const l of rawLinks) {
      if (typeNodes.has(l.source)) connected.add(l.target)
      if (typeNodes.has(l.target)) connected.add(l.source)
    }
    const fn = rawNodes.filter(n => connected.has(n.id))
    const fl = rawLinks.filter(l => connected.has(l.source) && connected.has(l.target))
    return { filteredNodes: fn, filteredLinks: fl }
  }, [rawNodes, rawLinks, mode, categoryId])

  const nodes = filteredNodes
  const links = filteredLinks
  const meta  = data?.meta ?? {}
  const types = [...new Set(nodes.map(n => n.type))]
  const errInfo = isError ? describeError(error) : null

  // Paper dropdown items
  const paperItems = (papers ?? []).map(p => ({ id: p.id, label: p.title }))
  // Topic dropdown items
  const topicItems = (topics ?? []).map(t => ({
    id: t.topic_id,
    label: `${t.label}  (${t.query_count} q · ${t.keyword_count} kw)`,
  }))
  // Category items
  const categoryItems = PREBUILT_TOPICS.map(c => ({ ...c }))

  // Counts per type
  const typeCounts = useMemo(() => {
    const m: Record<string, number> = {}
    for (const n of rawNodes) m[n.type] = (m[n.type] ?? 0) + 1
    return m
  }, [rawNodes])

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col p-4 md:p-6">

      {/* Header bar */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <FiGitBranch className="h-5 w-5 text-faint" />
          <div>
            <h1 className="font-display text-xl font-bold text-light">Knowledge Graph</h1>
            <div className="flex items-center gap-2 text-xs text-faint">
              <FiDatabase className="h-3 w-3" />
              <span>{rawNodes.length} nodes</span>
              <span>·</span>
              <span>{rawLinks.length} edges</span>
              {mode !== 'overview' && nodes.length !== rawNodes.length && (
                <span className="text-silver-light">({nodes.length} shown)</span>
              )}
              {isFetching && !isLoading && <span className="text-faint">refreshing…</span>}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Mode tabs */}
          {([
            ['overview', 'Overview'],
            ['paper',    'By Paper'],
            ['topic',    'By Topic'],
            ['category', 'By Type'],
          ] as [FilterMode, string][]).map(([m, lbl]) => (
            <button key={m}
              onClick={() => {
                setMode(m)
                setPaperId(''); setTopicId(''); setCategoryId('')
                if (m === 'overview') refetch()
              }}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                mode === m
                  ? 'bg-white/[0.08] text-light shadow-inner'
                  : 'text-faint hover:text-silver-light hover:bg-white/[0.03]'
              }`}>
              {lbl}
            </button>
          ))}

          <div className="mx-1 h-5 w-px bg-white/[0.06]" />

          <button onClick={() => refetch()} disabled={isFetching} aria-label="Refresh"
            className="rounded-lg border border-white/[0.06] p-2 text-faint hover:text-light disabled:opacity-30">
            <FiRefreshCw className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin' : ''}`} />
          </button>

          <button onClick={toggleFullScreen} aria-label={isFullScreen ? 'Exit full screen' : 'Full screen'}
            className="rounded-lg border border-white/[0.06] p-2 text-faint hover:text-light"
            title={isFullScreen ? 'Exit full screen (Esc)' : 'Full screen'}>
            {isFullScreen ? <FiMinimize2 className="h-3.5 w-3.5" /> : <FiMaximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Filter row */}
      <AnimatePresence mode="wait">
        {mode === 'paper' && (
          <motion.div key="paper" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mb-3 overflow-hidden">
            <SearchDropdown
              items={paperItems} value={paperId}
              onChange={setPaperId} placeholder="Search papers…"
            />
          </motion.div>
        )}
        {mode === 'topic' && (
          <motion.div key="topic" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mb-3 overflow-hidden">
            {topicItems.length > 0 ? (
              <SearchDropdown
                items={topicItems} value={topicId}
                onChange={setTopicId} placeholder="Search topics…"
              />
            ) : (
              <p className="py-2 text-xs text-faint">No topics yet. Ask a question in the chat first.</p>
            )}
          </motion.div>
        )}
        {mode === 'category' && (
          <motion.div key="cat" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mb-3 overflow-hidden">
            <div className="flex flex-wrap gap-2">
              {categoryItems.map(c => {
                const count = typeCounts[c.type] ?? 0
                const active = categoryId === c.id
                return (
                  <button key={c.id}
                    onClick={() => setCategoryId(active ? '' : c.id)}
                    className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-all ${
                      active
                        ? 'border-white/[0.15] bg-white/[0.08] text-light'
                        : 'border-white/[0.05] bg-white/[0.02] text-faint hover:bg-white/[0.05] hover:text-silver-light'
                    }`}>
                    <span>{c.icon}</span>
                    <span>{c.label}</span>
                    {count > 0 && <span className="rounded bg-white/[0.06] px-1.5 py-0.5 text-[9px]">{count}</span>}
                  </button>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Graph canvas area */}
      <div ref={graphWrapRef}
        className={`relative overflow-hidden bg-[#06060C] ${
          isFullScreen
            ? 'fixed inset-0 z-[100] rounded-none border-0'
            : 'flex-1 rounded-2xl border border-white/[0.06] bg-[#06060C]/70'
        }`}>

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center"><Spinner size={28} /></div>
        )}

        {isError && errInfo && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full border border-red-500/30 bg-red-500/10">
              <FiAlertCircle className="h-5 w-5 text-red-400" />
            </div>
            <p className="text-sm font-medium text-light">{errInfo.title}</p>
            <p className="max-w-sm text-xs text-dim">{errInfo.hint}</p>
            <button onClick={() => refetch()} className="rounded-xl bg-white/[0.06] px-5 py-2 text-sm text-light hover:bg-white/[0.10]">Retry</button>
          </div>
        )}

        {!isLoading && !isError && nodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center">
            <FiInfo className="h-6 w-6 text-faint" />
            <p className="text-sm text-dim">
              {mode === 'paper' && !paperId ? 'Select a paper above.' : ''}
              {mode === 'topic' && !topicId ? 'Select a topic above.' : ''}
              {mode === 'category' && !categoryId ? 'Select a category above.' : ''}
              {mode === 'overview' ? 'Graph is empty. Ingest a paper or ask a question.' : ''}
              {((mode === 'paper' && paperId) || (mode === 'topic' && topicId) || (mode === 'category' && categoryId)) && nodes.length === 0 ? 'No graph data for this selection.' : ''}
            </p>
          </div>
        )}

        {!isLoading && !isError && nodes.length > 0 && (
          <GraphScene nodes={nodes} links={links}
            fullScreen={isFullScreen}
            onToggleFullScreen={toggleFullScreen} />
        )}

        {/* Fullscreen exit hint */}
        {isFullScreen && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-lg border border-white/[0.08] bg-[#06060C]/80 px-3 py-1.5 text-[10px] text-faint backdrop-blur-xl">
            Press <kbd className="mx-0.5 rounded bg-white/[0.08] px-1.5 py-0.5 text-[9px] text-silver-light">Esc</kbd> or click ⛶ to exit full screen
          </div>
        )}

        {/* Legend */}
        {types.length > 0 && !isLoading && (
          <div className="absolute bottom-3 left-3 rounded-xl border border-white/[0.08] bg-[#06060C]/85 px-3 py-2.5 backdrop-blur-xl">
            <p className="mb-1.5 text-[9px] font-medium uppercase tracking-wider text-faint">Node types</p>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {types.map(t => (
                <span key={t} className="flex items-center gap-1.5 text-[10px] text-silver-light">
                  <span className="h-2 w-2 flex-shrink-0 rounded-full"
                    style={{ backgroundColor: TYPE_COLOR[t] ?? '#64748B', boxShadow: `0 0 6px ${TYPE_COLOR[t] ?? '#64748B'}50` }} />
                  {t} <span className="text-faint">({typeCounts[t] ?? 0})</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Stats badge */}
        {!isLoading && !isError && nodes.length > 0 && (
          <div className="absolute left-3 top-3 rounded-lg border border-white/[0.06] bg-[#06060C]/85 px-3 py-1.5 text-[10px] text-faint backdrop-blur-xl">
            {mode === 'category' && categoryId
              ? PREBUILT_TOPICS.find(c => c.id === categoryId)?.label ?? 'Filtered'
              : String(meta.filter ?? 'overview')} · {nodes.length}n · {links.length}e
          </div>
        )}

        {/* Navigation hint */}
        <div className="absolute bottom-3 right-3 hidden gap-3 text-[9px] text-faint md:flex">
          <span>Drag: rotate</span>
          <span>Shift+drag: pan</span>
          <span>Scroll: zoom</span>
          <span>Click: highlight</span>
        </div>
      </div>
    </div>
  )
}
