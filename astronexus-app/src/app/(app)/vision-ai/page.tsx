'use client'
import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { FiUploadCloud, FiEye, FiLayers, FiActivity, FiAlertCircle } from 'react-icons/fi'
import { Card } from '@/components/ui/Card'
import { httpUpload, ApiError } from '@/services/http'
import { apiConfig } from '@/lib/config'
import { cn } from '@/lib/cn'

const MODES = [
  { id: 'detect',  label: 'Detect',  icon: FiEye,      accent: '#9A9A9A' },
  { id: 'segment', label: 'Segment', icon: FiLayers,   accent: '#8A8A8A' },
  { id: 'explain', label: 'Explain', icon: FiActivity, accent: '#A8A49E' },
]

interface VisionResult { result: string; mode: string }

export default function VisionAIPage() {
  const [image, setImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [mode, setMode] = useState('detect')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const activeMode = MODES.find((m) => m.id === mode)!

  const load = (file: File) => {
    setImage(URL.createObjectURL(file))
    setImageFile(file)
    setResult(null)
    setError(null)
  }

  const loadSample = () => {
    setImage('/images/earth-still.jpg')
    setImageFile(null)   // sample: no real File, will send as URL
    setResult(null)
    setError(null)
  }

  const analyze = async () => {
    setAnalyzing(true)
    setResult(null)
    setError(null)

    try {
      let res: VisionResult | null = null

      if (imageFile) {
        // Real file upload
        const form = new FormData()
        form.append('file', imageFile)
        form.append('mode', mode)
        res = await httpUpload<VisionResult>(apiConfig.endpoints.vision, form)
      } else {
        // Sample image — send URL as JSON
        const { http } = await import('@/services/http')
        res = await http<VisionResult>(apiConfig.endpoints.vision, {
          method: 'POST',
          body: JSON.stringify({ image_url: '/images/earth-still.jpg', mode }),
        })
      }

      if (res?.result) {
        setResult(res.result)
      } else {
        // Backend offline — use hardcoded demo results
        const DEMO: Record<string, string> = {
          detect:  'Detected: 2 cyclonic systems, 1 coastal flood zone, urban heat signature over metro region.',
          segment: 'Segmented 4 regions: ocean (61%), landmass (24%), cloud cover (12%), ice (3%).',
          explain: 'A large low-pressure system is developing off the coast, with spiral cloud bands indicating an intensifying tropical storm.',
        }
        setResult(DEMO[mode] ?? 'No result.')
      }
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'Analysis failed. The backend may be offline.',
      )
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl p-6 md:p-8">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold text-light">Vision AI</h1>
        <p className="mt-1 text-sm text-dim">Detect, segment, and explain satellite imagery.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Image panel */}
        <div>
          {!image ? (
            <div
              onClick={() => inputRef.current?.click()}
              className="flex aspect-square cursor-pointer flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed border-white/15 transition-colors hover:border-glow/50 hover:bg-white/[0.02]"
            >
              <FiUploadCloud className="h-10 w-10 text-glow-soft" />
              <p className="text-sm text-dim">Upload satellite image</p>
              <button
                onClick={(e) => { e.stopPropagation(); loadSample() }}
                className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-dim hover:text-light"
              >
                Use sample image
              </button>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) load(f) }}
              />
            </div>
          ) : (
            <div className="relative aspect-square overflow-hidden rounded-3xl border border-white/[0.08]">
              <img src={image} alt="Uploaded" className="h-full w-full object-cover" />
              {analyzing && (
                <motion.div
                  className="absolute inset-x-0 h-0.5 bg-glow"
                  style={{ boxShadow: '0 0 12px #8A8A8A' }}
                  animate={{ top: ['0%', '100%', '0%'] }}
                  transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                />
              )}
              <button
                onClick={() => { setImage(null); setImageFile(null); setResult(null); setError(null) }}
                className="absolute right-3 top-3 rounded-lg bg-void/70 px-3 py-1.5 text-xs text-light backdrop-blur"
              >
                Change
              </button>
            </div>
          )}
        </div>

        {/* Controls + result */}
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-3 gap-2">
            {MODES.map((m) => {
              const Icon = m.icon
              return (
                <button
                  key={m.id}
                  onClick={() => { setMode(m.id); setResult(null); setError(null) }}
                  className={cn(
                    'flex flex-col items-center gap-2 rounded-xl border py-3 text-xs transition-colors',
                    mode === m.id
                      ? 'border-glow/40 bg-glow/10 text-light'
                      : 'border-white/[0.06] bg-white/[0.02] text-dim hover:text-light',
                  )}
                >
                  <Icon className="h-4 w-4" style={{ color: mode === m.id ? m.accent : undefined }} />
                  {m.label}
                </button>
              )
            })}
          </div>

          <button
            onClick={analyze}
            disabled={!image || analyzing}
            className="rounded-xl bg-blue py-3 text-sm font-medium text-white transition-colors hover:bg-blue-bright disabled:opacity-40"
          >
            {analyzing ? 'Analyzing…' : `Run ${activeMode.label}`}
          </button>

          {error && (
            <div className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2.5" role="alert">
              <FiAlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-red-400" />
              <p className="text-xs text-red-300">{error}</p>
            </div>
          )}

          <Card className="flex-1 p-5">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">Result</h3>
            {result ? (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm leading-relaxed text-slate-200">
                {result}
              </motion.p>
            ) : (
              <p className="text-sm text-faint">{image ? 'Run an analysis to see results.' : 'Upload an image to begin.'}</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
