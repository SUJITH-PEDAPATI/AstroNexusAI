'use client'
import Link from 'next/link'
import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FiFileText, FiMessageSquare, FiClock, FiRefreshCw,
  FiAlertCircle, FiTrash2, FiLoader,
} from 'react-icons/fi'
import { Card }     from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn }       from '@/lib/cn'
import { http, httpDelete, ApiError } from '@/services/http'
import { apiConfig } from '@/lib/config'
import type { Paper } from '@/types'

// ── Status badge colours ───────────────────────────────────────────────────────
const STATUS: Record<Paper['status'], string> = {
  ready:      'bg-white/[0.06] text-silver-light',
  processing: 'bg-white/[0.05] text-silver',
  failed:     'bg-red-500/12 text-red-300',
}

// ── Error → human-readable message ────────────────────────────────────────────
function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return 'Endpoint not found (404). Is the backend updated?'
    if (err.status === 503) return 'Neo4j is unavailable (503). Check the graph database.'
    if (err.status >= 500) return `Backend error ${err.status}: ${err.message}`
    if (err.status === 401 || err.status === 403) return `Authentication error (${err.status}).`
    return `API error ${err.status}: ${err.message}`
  }
  if (err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('network')))
    return `Cannot reach the backend. Check that the server is running on ${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}.`
  if (err instanceof Error && err.message.toLowerCase().includes('cors'))
    return 'CORS blocked the request. Check the backend CORS configuration.'
  if (err instanceof Error) return err.message
  return 'Unknown error. Check the browser console for details.'
}

// ── Clear-papers response shape ────────────────────────────────────────────────
interface ClearResult {
  success:        boolean
  message:        string
  deleted_papers: number
  deleted_chunks: number
}

// ── Page ───────────────────────────────────────────────────────────────────────
export default function PapersPage() {
  const qc = useQueryClient()

  // Paper list query
  const { data: papers, isLoading, isError, error, refetch } = useQuery<Paper[], Error>({
    queryKey: ['papers'],
    queryFn: async () => {
      const res = await http<Paper[]>(apiConfig.endpoints.papers)
      if (res === null)
        throw new Error(
          `Cannot reach the backend. Check that the server is running on ${
            process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}.`
        )
      return res
    },
    staleTime: 30_000,
    retry: 1,
  })

  // Clear-all state
  const [showConfirm, setShowConfirm] = useState(false)
  const [clearing,    setClearing]    = useState(false)
  const [clearNote,   setClearNote]   = useState<{ ok: boolean; msg: string } | null>(null)

  const handleClear = async () => {
    setShowConfirm(false)
    setClearing(true)
    setClearNote(null)
    try {
      const res = await httpDelete<ClearResult>(apiConfig.endpoints.clearPapers)
      if (res?.success) {
        setClearNote({
          ok:  true,
          msg: res.deleted_papers > 0
            ? `Cleared ${res.deleted_papers} paper${res.deleted_papers !== 1 ? 's' : ''} and ${res.deleted_chunks} chunks.`
            : 'Knowledge base was already empty.',
        })
        // Invalidate both caches so cards disappear and dashboard updates
        qc.invalidateQueries({ queryKey: ['papers'] })
        qc.invalidateQueries({ queryKey: ['dashboard-stats'] })
      } else {
        setClearNote({ ok: false, msg: res?.message ?? 'Clear failed. Check server logs.' })
      }
    } catch (e) {
      setClearNote({
        ok:  false,
        msg: e instanceof ApiError
          ? `Server error ${e.status}: ${e.message}`
          : 'Could not reach the backend.',
      })
    } finally {
      setClearing(false)
    }
  }

  const errorMessage = isError ? describeError(error) : null
  const hasPapers    = !isLoading && !isError && (papers?.length ?? 0) > 0

  return (
    <div className="mx-auto max-w-5xl p-6 md:p-8">

      {/* Header ────────────────────────────────────────────────────────────── */}
      <div className="mb-8 flex items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Papers</h1>
          <p className="mt-1 text-sm text-dim">Your ingested research corpus.</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            aria-label="Refresh papers"
            className="rounded-xl border border-white/10 px-4 py-2.5 text-sm text-dim transition-colors hover:text-light disabled:opacity-40"
          >
            <FiRefreshCw className="h-4 w-4" />
          </button>

          {/* Clear All Papers — only shown when papers exist or after a clear note */}
          <button
            onClick={() => { setClearNote(null); setShowConfirm(true) }}
            disabled={clearing || isLoading}
            aria-label="Clear all ingested papers"
            className="flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm font-medium text-red-400 transition-colors hover:bg-red-500/18 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {clearing
              ? <motion.span animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}><FiLoader className="h-4 w-4" /></motion.span>
              : <FiTrash2 className="h-4 w-4" />}
            {clearing ? 'Clearing…' : 'Clear All Papers'}
          </button>

          <Link
            href="/research"
            className="rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-bright"
          >
            Ingest paper
          </Link>
        </div>
      </div>

      {/* Clear operation notification ────────────────────────────────────────── */}
      <AnimatePresence>
        {clearNote && (
          <motion.div
            key="note"
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className={cn(
              'mb-6 flex items-start gap-2.5 rounded-xl border px-4 py-3',
              clearNote.ok
                ? 'border-white/[0.10] bg-white/[0.04]'
                : 'border-red-500/30 bg-red-500/10',
            )}
          >
            {clearNote.ok
              ? <FiFileText className="mt-0.5 h-4 w-4 flex-shrink-0 text-silver-bright" />
              : <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />}
            <p className={cn('text-sm', clearNote.ok ? 'text-silver-light' : 'text-red-300')}>
              {clearNote.msg}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading skeletons ───────────────────────────────────────────────────── */}
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-2xl" />
          ))}
        </div>
      )}

      {/* Error state ─────────────────────────────────────────────────────────── */}
      {isError && !isLoading && (
        <Card className="p-8">
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full border border-red-500/25 bg-red-500/10">
              <FiAlertCircle className="h-5 w-5 text-red-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-light">Failed to load papers</p>
              <p className="mt-1 max-w-sm text-xs leading-relaxed text-dim">{errorMessage}</p>
            </div>
            <button
              onClick={() => refetch()}
              className="rounded-xl bg-white/[0.06] px-5 py-2 text-sm font-medium text-light transition-colors hover:bg-white/[0.10]"
            >
              Retry
            </button>
          </div>
        </Card>
      )}

      {/* Empty state ─────────────────────────────────────────────────────────── */}
      {!isLoading && !isError && papers?.length === 0 && (
        <Card className="p-12 text-center">
          <FiFileText className="mx-auto h-8 w-8 text-faint" />
          <p className="mt-3 text-sm font-medium text-light">No papers ingested yet.</p>
          <p className="mt-1 text-xs text-dim">Upload a research paper to start building your corpus.</p>
          <Link
            href="/research"
            className="mt-4 inline-block rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-bright"
          >
            Ingest your first paper
          </Link>
        </Card>
      )}

      {/* Paper grid ──────────────────────────────────────────────────────────── */}
      {!isLoading && !isError && papers && papers.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2">
          {papers.map((p) => (
            <Card key={p.id} hover className="flex flex-col p-5">
              <div className="mb-3 flex items-start justify-between gap-3">
                <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-blue/10 text-glow-soft">
                  <FiFileText className="h-5 w-5" />
                </span>
                <span className={cn('rounded-full px-2.5 py-1 text-[10px] font-medium capitalize', STATUS[p.status])}>
                  {p.status}
                </span>
              </div>
              <h3 className="font-display text-sm font-semibold leading-snug text-light">{p.title}</h3>
              <p className="mt-1 text-xs text-dim">{p.authors?.join(', ')}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {p.keywords?.map((k) => (
                  <span key={k} className="rounded-md bg-white/[0.04] px-2 py-0.5 text-[10px] text-dim">{k}</span>
                ))}
              </div>
              <div className="mt-auto flex items-center justify-between pt-4">
                <span className="flex items-center gap-1 text-[10px] text-faint">
                  <FiClock className="h-3 w-3" />{p.chunks ?? 0} chunks
                </span>
                {p.status === 'ready' && (
                  <Link
                    href="/chat"
                    className="flex items-center gap-1.5 rounded-lg bg-blue/15 px-3 py-1.5 text-xs font-medium text-glow-soft transition-colors hover:bg-blue/25"
                  >
                    <FiMessageSquare className="h-3 w-3" /> Query
                  </Link>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Confirmation modal ───────────────────────────────────────────────────── */}
      <AnimatePresence>
        {showConfirm && (
          <>
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
              onClick={() => setShowConfirm(false)}
            />
            <motion.div
              key="dialog"
              initial={{ opacity: 0, scale: 0.96, y: 8 }}
              animate={{ opacity: 1, scale: 1,    y: 0 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.2 }}
              className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-white/[0.08] bg-[#0C0C0C] p-6 shadow-2xl"
              role="alertdialog" aria-modal="true" aria-labelledby="clear-title"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full border border-red-500/30 bg-red-500/10">
                <FiTrash2 className="h-5 w-5 text-red-400" />
              </div>
              <h2 id="clear-title" className="mt-4 font-display text-base font-semibold text-light">
                Clear all papers?
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-dim">
                This will permanently remove all ingested papers, their embeddings,
                chunks, metadata, and stored files from the knowledge base.
                This action cannot be undone.
              </p>
              <p className="mt-2 text-xs text-faint">
                {hasPapers ? `${papers!.length} paper${papers!.length !== 1 ? 's' : ''} will be deleted.` : ''}
              </p>
              <div className="mt-6 flex gap-3">
                <button
                  onClick={() => setShowConfirm(false)}
                  className="flex-1 rounded-xl border border-white/[0.10] bg-white/[0.04] py-2.5 text-sm font-medium text-silver transition-colors hover:bg-white/[0.07]"
                >
                  Cancel
                </button>
                <button
                  onClick={handleClear}
                  className="flex-1 rounded-xl border border-red-500/40 bg-red-500/15 py-2.5 text-sm font-medium text-red-400 transition-colors hover:bg-red-500/25"
                >
                  Clear All Papers
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

    </div>
  )
}
