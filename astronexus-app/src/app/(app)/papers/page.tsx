'use client'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { FiFileText, FiMessageSquare, FiClock, FiRefreshCw } from 'react-icons/fi'
import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn } from '@/lib/cn'
import { http } from '@/services/http'
import { apiConfig } from '@/lib/config'
import type { Paper } from '@/types'

const STATUS: Record<Paper['status'], string> = {
  ready: 'bg-white/[0.06] text-silver-light',
  processing: 'bg-white/[0.05] text-silver',
  failed: 'bg-red-500/12 text-red-300',
}

export default function PapersPage() {
  const { data: papers, isLoading, isError, refetch } = useQuery<Paper[]>({
    queryKey: ['papers'],
    queryFn: async () => {
      const res = await http<Paper[]>(apiConfig.endpoints.papers)
      return res ?? []
    },
    staleTime: 30_000,
  })

  return (
    <div className="mx-auto max-w-5xl p-6 md:p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Papers</h1>
          <p className="mt-1 text-sm text-dim">Your ingested research corpus.</p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => refetch()}
            className="rounded-xl border border-white/10 px-4 py-2.5 text-sm text-dim transition-colors hover:text-light"
            aria-label="Refresh">
            <FiRefreshCw className="h-4 w-4" />
          </button>
          <Link href="/research" className="rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-bright">
            Ingest paper
          </Link>
        </div>
      </div>

      {/* Loading skeletons */}
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-2xl" />
          ))}
        </div>
      )}

      {/* Error state */}
      {isError && !isLoading && (
        <Card className="p-8 text-center">
          <p className="text-sm text-dim">Could not load papers — backend may be offline.</p>
          <button onClick={() => refetch()} className="mt-4 rounded-xl bg-blue px-4 py-2 text-sm text-white">Retry</button>
        </Card>
      )}

      {/* Empty state */}
      {!isLoading && !isError && papers?.length === 0 && (
        <Card className="p-12 text-center">
          <FiFileText className="mx-auto h-8 w-8 text-faint" />
          <p className="mt-3 text-sm text-dim">No papers ingested yet.</p>
          <Link href="/research" className="mt-4 inline-block rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white">
            Ingest your first paper
          </Link>
        </Card>
      )}

      {/* Paper grid */}
      {!isLoading && papers && papers.length > 0 && (
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
                  <Link href="/chat"
                    className="flex items-center gap-1.5 rounded-lg bg-blue/15 px-3 py-1.5 text-xs font-medium text-glow-soft transition-colors hover:bg-blue/25">
                    <FiMessageSquare className="h-3 w-3" /> Query
                  </Link>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
