'use client'
import { useState } from 'react'
import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { FiSearch, FiStar, FiMessageSquare, FiTrash2, FiFilter } from 'react-icons/fi'
import { useChatStore } from '@/store/chat.store'
import { Card } from '@/components/ui/Card'
import { cn } from '@/lib/cn'

type Filter = 'all' | 'favorites' | 'pinned'

export default function HistoryPage() {
  const { conversations, setActive, remove, toggleFavorite, togglePin } = useChatStore()
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')

  const filtered = conversations
    .filter((c) => c.title.toLowerCase().includes(query.toLowerCase()))
    .filter((c) => (filter === 'favorites' ? c.favorite : filter === 'pinned' ? c.pinned : true))
    .sort((a, b) => b.updatedAt - a.updatedAt)

  return (
    <div className="mx-auto max-w-4xl p-6 md:p-8">
      <div className="mb-6">
        <h1 className="font-display text-2xl font-bold text-light">Chat History</h1>
        <p className="mt-1 text-sm text-dim">Every conversation, searchable and organized.</p>
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2.5">
          <FiSearch className="h-4 w-4 text-faint" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search conversations…"
            className="flex-1 bg-transparent text-sm text-light outline-none placeholder:text-faint" />
        </div>
        <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1">
          <FiFilter className="ml-2 h-3.5 w-3.5 text-faint" />
          {(['all', 'favorites', 'pinned'] as Filter[]).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              className={cn('rounded-lg px-3 py-1.5 text-xs capitalize transition-colors', filter === f ? 'bg-blue/20 text-light' : 'text-dim hover:text-light')}>{f}</button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <Card className="p-12 text-center">
          <FiMessageSquare className="mx-auto h-8 w-8 text-faint" />
          <p className="mt-3 text-sm text-dim">No conversations found.</p>
          <Link href="/chat" className="mt-4 inline-block rounded-xl bg-blue px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-bright">Start chatting</Link>
        </Card>
      ) : (
        <div className="space-y-2">
          {filtered.map((c) => (
            <Card key={c.id} hover className="group flex items-center gap-4 p-4">
              <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-blue/10 text-glow-soft"><FiMessageSquare className="h-4 w-4" /></span>
              <Link href="/chat" onClick={() => setActive(c.id)} className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-light">{c.title}</div>
                <div className="mt-0.5 text-xs text-faint">
                  {c.messages.length} messages · {formatDistanceToNow(c.updatedAt, { addSuffix: true })}
                  {c.folder && <span className="ml-2 rounded bg-white/5 px-1.5 py-0.5">{c.folder}</span>}
                </div>
              </Link>
              <div className="flex flex-shrink-0 items-center gap-1">
                <button onClick={() => togglePin(c.id)} aria-label="Pin" className={cn('rounded-lg p-2 transition-colors', c.pinned ? 'text-glow-soft' : 'text-faint hover:text-light')}>
                  <FiMessageSquare className="h-3.5 w-3.5" />
                </button>
                <button onClick={() => toggleFavorite(c.id)} aria-label="Favorite" className="rounded-lg p-2 text-faint transition-colors hover:text-gold">
                  <FiStar className={cn('h-3.5 w-3.5', c.favorite && 'fill-gold text-gold')} />
                </button>
                <button onClick={() => remove(c.id)} aria-label="Delete" className="rounded-lg p-2 text-faint transition-colors hover:text-red-400"><FiTrash2 className="h-3.5 w-3.5" /></button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
