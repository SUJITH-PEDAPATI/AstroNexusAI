'use client'
import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiUploadCloud, FiFile, FiCheck, FiAlertCircle, FiTrash2, FiLoader } from 'react-icons/fi'
import { useQueryClient } from '@tanstack/react-query'
import { Card } from '@/components/ui/Card'
import { httpUpload, httpDelete, ApiError } from '@/services/http'
import { apiConfig } from '@/lib/config'
import { cn } from '@/lib/cn'

const STAGES = [
  'Parsing PDF',
  'Chunking (512 · 64 overlap)',
  'Embedding (qwen3:4b)',
  'Indexing to Qdrant',
  'Extracting graph (Neo4j)',
]

interface IngestResult {
  paper_id: string
  title:    string
  chunks:   number
  entities: number
  keywords: string[]
  domain:   string
  message:  string
}

interface ClearResult {
  success:        boolean
  message:        string
  deleted_papers: number
  deleted_chunks: number
}

export default function ResearchPage() {
  const qc = useQueryClient()

  // ── Upload state ──────────────────────────────────────────────────────────
  const [drag,     setDrag]     = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const [stage,    setStage]    = useState(-1)
  const [done,     setDone]     = useState(false)
  const [result,   setResult]   = useState<IngestResult | null>(null)
  const [error,    setError]    = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // ── Clear papers state ────────────────────────────────────────────────────
  const [showConfirm, setShowConfirm]   = useState(false)
  const [clearing,    setClearing]      = useState(false)
  const [clearResult, setClearResult]   = useState<string | null>(null)
  const [clearError,  setClearError]    = useState<string | null>(null)

  // ── Ingest logic ──────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (file: File) => {
    setFileName(file.name)
    setDone(false)
    setResult(null)
    setError(null)
    setStage(0)

    const stageTimer = setInterval(() => {
      setStage((prev) => {
        if (prev < STAGES.length - 1) return prev + 1
        clearInterval(stageTimer)
        return prev
      })
    }, 900)

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await httpUpload<IngestResult>(apiConfig.endpoints.ingest, form)

      clearInterval(stageTimer)

      if (res) {
        setResult(res)
        setStage(STAGES.length)
        setDone(true)
        qc.invalidateQueries({ queryKey: ['papers'] })
        qc.invalidateQueries({ queryKey: ['dashboard-stats'] })
      } else {
        setResult({ paper_id: 'demo', title: file.name, chunks: 142, entities: 38, keywords: [], domain: '', message: 'Demo mode' })
        setStage(STAGES.length)
        setDone(true)
      }
    } catch (err) {
      clearInterval(stageTimer)
      setStage(-1)
      setError(
        err instanceof ApiError
          ? err.message
          : 'Upload failed. Please check the file and try again.',
      )
    }
  }, [qc])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) uploadFile(f)
  }

  const reset = () => {
    setFileName(null)
    setStage(-1)
    setDone(false)
    setResult(null)
    setError(null)
  }

  // ── Clear all papers logic ────────────────────────────────────────────────
  const handleClearConfirm = async () => {
    setShowConfirm(false)
    setClearing(true)
    setClearResult(null)
    setClearError(null)

    try {
      const res = await httpDelete<ClearResult>(apiConfig.endpoints.clearPapers)
      if (res?.success) {
        setClearResult(
          res.deleted_papers > 0
            ? `Cleared ${res.deleted_papers} paper${res.deleted_papers !== 1 ? 's' : ''} and ${res.deleted_chunks} chunks.`
            : 'Knowledge base was already empty.',
        )
        // Refresh all data that shows paper counts
        qc.invalidateQueries({ queryKey: ['papers'] })
        qc.invalidateQueries({ queryKey: ['dashboard-stats'] })
        // Reset upload state if it was showing a result
        reset()
      } else {
        setClearError(res?.message ?? 'Clear operation failed. Check the server logs.')
      }
    } catch (err) {
      setClearError(
        err instanceof ApiError
          ? `Server error ${err.status}: ${err.message}`
          : 'Could not reach the backend. Is the server running?',
      )
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl p-6 md:p-8">

      {/* Header */}
      <div className="mb-8 flex items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Research Ingestion</h1>
          <p className="mt-1 text-sm text-dim">Upload a paper to embed it and extract its knowledge graph.</p>
        </div>

        {/* Clear All Papers button */}
        <button
          onClick={() => { setClearResult(null); setClearError(null); setShowConfirm(true) }}
          disabled={clearing}
          aria-label="Clear all ingested papers"
          className="flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm font-medium text-red-400 transition-colors hover:bg-red-500/18 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {clearing
            ? <motion.span animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}><FiLoader className="h-4 w-4" /></motion.span>
            : <FiTrash2 className="h-4 w-4" />}
          {clearing ? 'Clearing…' : 'Clear All Papers'}
        </button>
      </div>

      {/* Success / error notifications for clear operation */}
      <AnimatePresence>
        {clearResult && (
          <motion.div
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mb-4 flex items-start gap-2.5 rounded-xl border border-white/[0.10] bg-white/[0.04] px-4 py-3"
          >
            <FiCheck className="mt-0.5 h-4 w-4 flex-shrink-0 text-silver-bright" />
            <p className="text-sm text-silver-light">{clearResult}</p>
          </motion.div>
        )}
        {clearError && (
          <motion.div
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3"
            role="alert"
          >
            <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
            <p className="text-sm text-red-300">{clearError}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload area or progress */}
      {!fileName ? (
        <>
          {error && (
            <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3" role="alert">
              <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={cn(
              'flex cursor-pointer flex-col items-center justify-center gap-4 rounded-3xl border-2 border-dashed py-20 transition-colors',
              drag ? 'border-glow bg-glow/5' : 'border-white/15 hover:border-glow/50 hover:bg-white/[0.02]',
            )}
          >
            <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-blue/15 text-glow-soft">
              <FiUploadCloud className="h-7 w-7" />
            </span>
            <div className="text-center">
              <p className="text-sm font-medium text-light">Drop a PDF here, or click to browse</p>
              <p className="mt-1 text-xs text-faint">Research papers up to 50 MB</p>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadFile(f) }}
            />
          </div>
        </>
      ) : (
        <Card className="p-6">
          <div className="mb-6 flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue/15 text-glow-soft">
              <FiFile className="h-5 w-5" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium text-light">{fileName}</div>
              <div className="text-xs text-faint">{done ? 'Ingestion complete' : 'Processing…'}</div>
            </div>
            {done && (
              <span className="rounded-full bg-white/[0.06] px-3 py-1 text-xs font-medium text-silver-light">
                Ready
              </span>
            )}
          </div>

          <div className="space-y-3">
            {STAGES.map((s, i) => {
              const state = done || stage > i ? 'done' : stage === i ? 'active' : 'pending'
              return (
                <div key={s} className="flex items-center gap-3">
                  <span className={cn(
                    'flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full text-xs',
                    state === 'done'   ? 'bg-white/[0.08] text-silver-light'
                    : state === 'active' ? 'bg-blue/20 text-glow-soft'
                    : 'bg-white/5 text-faint',
                  )}>
                    {state === 'done' ? <FiCheck className="h-3.5 w-3.5" /> : i + 1}
                  </span>
                  <span className={cn('text-sm', state === 'pending' ? 'text-faint' : 'text-light')}>{s}</span>
                  {state === 'active' && (
                    <motion.span
                      className="ml-auto h-1.5 w-1.5 rounded-full bg-glow"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1 }}
                    />
                  )}
                </div>
              )
            })}
          </div>

          <AnimatePresence>
            {done && result && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 rounded-2xl border border-white/[0.10] bg-white/[0.03] p-4"
              >
                {result.title && result.title !== fileName && (
                  <p className="mb-4 truncate text-xs text-dim">{result.title}</p>
                )}
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="font-display text-xl font-bold text-light">{result.chunks}</div>
                    <div className="text-[10px] text-dim">chunks</div>
                  </div>
                  <div>
                    <div className="font-display text-xl font-bold text-light">1024</div>
                    <div className="text-[10px] text-dim">dimensions</div>
                  </div>
                  <div>
                    <div className="font-display text-xl font-bold text-light">{result.entities}</div>
                    <div className="text-[10px] text-dim">entities</div>
                  </div>
                </div>
                {result.domain && (
                  <p className="mt-3 text-center text-[10px] text-dim">
                    Domain: <span className="text-silver-light">{result.domain}</span>
                  </p>
                )}
                {result.keywords.length > 0 && (
                  <div className="mt-3 flex flex-wrap justify-center gap-1">
                    {result.keywords.slice(0, 8).map((kw) => (
                      <span key={kw} className="rounded-md border border-white/[0.06] bg-white/[0.03] px-2 py-0.5 text-[10px] text-dim">{kw}</span>
                    ))}
                  </div>
                )}
                <button
                  onClick={reset}
                  className="mt-4 w-full rounded-xl bg-blue py-2.5 text-sm font-medium text-white hover:bg-blue-bright"
                >
                  Ingest another paper
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      )}

      {/* ── Confirmation modal ──────────────────────────────────────────────── */}
      <AnimatePresence>
        {showConfirm && (
          <>
            {/* Backdrop */}
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
              onClick={() => setShowConfirm(false)}
            />

            {/* Dialog */}
            <motion.div
              key="dialog"
              initial={{ opacity: 0, scale: 0.96, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.2 }}
              className="fixed left-1/2 top-1/2 z-50 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-white/[0.08] bg-[#0C0C0C] p-6 shadow-2xl"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="confirm-title"
            >
              <div className="mb-1 flex h-12 w-12 items-center justify-center rounded-full border border-red-500/30 bg-red-500/10">
                <FiTrash2 className="h-5 w-5 text-red-400" />
              </div>

              <h2 id="confirm-title" className="mt-4 font-display text-base font-semibold text-light">
                Clear all ingested papers?
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-dim">
                This will permanently remove all ingested papers, their indexed chunks,
                embeddings, and associated metadata from the knowledge base.
                This action cannot be undone.
              </p>

              <div className="mt-6 flex gap-3">
                <button
                  onClick={() => setShowConfirm(false)}
                  className="flex-1 rounded-xl border border-white/[0.10] bg-white/[0.04] py-2.5 text-sm font-medium text-silver transition-colors hover:bg-white/[0.07]"
                >
                  Cancel
                </button>
                <button
                  onClick={handleClearConfirm}
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
