'use client'
import Link from 'next/link'
import { FiFileText, FiMessageSquare, FiClock } from 'react-icons/fi'
import { Card } from '@/components/ui/Card'
import { cn } from '@/lib/cn'
import type { Paper } from '@/types'

const PAPERS: Paper[] = [
  { id: 'p1', title: 'The Origins Space Telescope: Mission Concept', authors: ['A. Pope', 'J. Bauer'], chunks: 142, status: 'ready', uploadedAt: Date.now() - 3600e3, keywords: ['infrared', 'exoplanets', 'spectroscopy'] },
  { id: 'p2', title: 'Exoplanet Atmospheric Biosignatures in the IR', authors: ['K. Pontoppidan'], chunks: 98, status: 'ready', uploadedAt: Date.now() - 8400e3, keywords: ['biosignatures', 'atmospheres'] },
  { id: 'p3', title: 'SAR Flood Detection with Deep Learning', authors: ['M. Rivera', 'T. Cho'], chunks: 76, status: 'ready', uploadedAt: Date.now() - 172800e3, keywords: ['SAR', 'remote sensing', 'floods'] },
  { id: 'p4', title: 'Galaxy Evolution Across Cosmic Time', authors: ['L. Armus'], chunks: 0, status: 'processing', uploadedAt: Date.now() - 120e3, keywords: ['galaxies'] },
]

const STATUS: Record<Paper['status'], string> = {
  ready: 'bg-emerald-500/15 text-emerald-400',
  processing: 'bg-blue/15 text-glow-soft',
  failed: 'bg-red-500/15 text-red-400',
}

export default function PapersPage() {
  return (
    <div className="mx-auto max-w-5xl p-6 md:p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Papers</h1>
          <p className="mt-1 text-sm text-dim">Your ingested research corpus.</p>
        </div>
        <Link href="/research" className="rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-bright">Ingest paper</Link>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {PAPERS.map((p) => (
          <Card key={p.id} hover className="flex flex-col p-5">
            <div className="mb-3 flex items-start justify-between gap-3">
              <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-blue/10 text-glow-soft"><FiFileText className="h-5 w-5" /></span>
              <span className={cn('rounded-full px-2.5 py-1 text-[10px] font-medium capitalize', STATUS[p.status])}>{p.status}</span>
            </div>
            <h3 className="font-display text-sm font-semibold leading-snug text-light">{p.title}</h3>
            <p className="mt-1 text-xs text-dim">{p.authors.join(', ')}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {p.keywords.map((k) => <span key={k} className="rounded-md bg-white/[0.04] px-2 py-0.5 text-[10px] text-dim">{k}</span>)}
            </div>
            <div className="mt-auto flex items-center justify-between pt-4">
              <span className="flex items-center gap-1 text-[10px] text-faint"><FiClock className="h-3 w-3" />{p.chunks} chunks</span>
              {p.status === 'ready' && (
                <Link href="/chat" className="flex items-center gap-1.5 rounded-lg bg-blue/15 px-3 py-1.5 text-xs font-medium text-glow-soft transition-colors hover:bg-blue/25">
                  <FiMessageSquare className="h-3 w-3" /> Query
                </Link>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
