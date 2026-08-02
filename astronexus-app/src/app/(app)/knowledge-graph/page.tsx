'use client'
import dynamic from 'next/dynamic'
import { FiMove, FiZoomIn, FiMousePointer } from 'react-icons/fi'
import { Spinner } from '@/components/ui/Spinner'
import { GRAPH_NODES } from '@/features/knowledge-graph/data'

const GraphScene = dynamic(() => import('@/features/knowledge-graph/GraphScene').then((m) => m.GraphScene), {
  ssr: false,
  loading: () => <div className="flex h-full items-center justify-center"><Spinner size={28} /></div>,
})

const TYPES = [...new Set(GRAPH_NODES.map((n) => n.type))]
const TYPE_COLOR: Record<string, string> = {
  Paper: '#8A8A8A', Author: '#9A9A9A', Keyword: '#B4B4B4', Domain: '#D8D8D8', Institution: '#7E7E7E', Satellite: '#9A9A9A',
}

export default function KnowledgeGraphPage() {
  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col p-6 md:p-8">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Knowledge Graph</h1>
          <p className="mt-1 text-sm text-dim">Explore how entities connect across your corpus.</p>
        </div>
        <div className="hidden items-center gap-4 text-xs text-dim md:flex">
          <span className="flex items-center gap-1.5"><FiMove className="h-3.5 w-3.5" /> Drag to rotate</span>
          <span className="flex items-center gap-1.5"><FiZoomIn className="h-3.5 w-3.5" /> Scroll to zoom</span>
          <span className="flex items-center gap-1.5"><FiMousePointer className="h-3.5 w-3.5" /> Click to expand</span>
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden rounded-3xl border border-white/[0.07] bg-space/40">
        <GraphScene />
        {/* Legend */}
        <div className="absolute bottom-4 left-4 flex flex-wrap gap-3 rounded-2xl border border-white/10 bg-void/70 px-4 py-3 backdrop-blur-xl">
          {TYPES.map((t) => (
            <span key={t} className="flex items-center gap-1.5 text-[11px] text-dim">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: TYPE_COLOR[t], boxShadow: `0 0 8px ${TYPE_COLOR[t]}` }} />{t}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
