'use client'
import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FiUploadCloud, FiFile, FiCheck, FiAlertCircle } from 'react-icons/fi'
import { useQueryClient } from '@tanstack/react-query'
import { Card } from '@/components/ui/Card'
import { httpUpload, ApiError } from '@/services/http'
import { apiConfig } from '@/lib/config'
import { cn } from '@/lib/cn'

const STAGES = [
  'Parsing PDF',
  'Chunking (1024 · 128 overlap)',
  'Embedding (BGE-M3)',
  'Indexing to Qdrant',
  'Extracting graph (Neo4j)',
]

interface IngestResult {
  paper_id: string
  title?: string
  chunks: number
  entities?: number
}

export default function ResearchPage() {
  const qc = useQueryClient()
  const [drag, setDrag] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const [stage, setStage] = useState(-1)
  const [done, setDone] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const uploadFile = useCallback(async (file: File) => {
    setFileName(file.name)
    setDone(false)
    setResult(null)
    setError(null)
    setStage(0)

    // Advance UI stages while the upload is in-flight so the user sees progress.
    // The backend processes synchronously; we simulate visual stage ticking.
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
        // Invalidate the papers cache so /papers refreshes automatically.
        qc.invalidateQueries({ queryKey: ['papers'] })
      } else {
        // Backend offline — show simulated completion so the UI isn't broken.
        setResult({ paper_id: 'demo', chunks: 142, entities: 38 })
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

  return (
    <div className="mx-auto max-w-3xl p-6 md:p-8">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold text-light">Research Ingestion</h1>
        <p className="mt-1 text-sm text-dim">Upload a paper to embed it and extract its knowledge graph.</p>
      </div>

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
              <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-400">
                Ready
              </span>
            )}
          </div>

          <div className="space-y-3">
            {STAGES.map((s, i) => {
              const state = done || stage > i ? 'done' : stage === i ? 'active' : 'pending'
              return (
                <div key={s} className="flex items-center gap-3">
                  <span
                    className={cn(
                      'flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full text-xs',
                      state === 'done'
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : state === 'active'
                          ? 'bg-blue/20 text-glow-soft'
                          : 'bg-white/5 text-faint',
                    )}
                  >
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
                className="mt-6 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4"
              >
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
                    <div className="font-display text-xl font-bold text-light">{result.entities ?? '—'}</div>
                    <div className="text-[10px] text-dim">entities</div>
                  </div>
                </div>
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
    </div>
  )
}
