'use client'
import { useState } from 'react'
import { FiPlus, FiSearch, FiMessageSquare, FiStar, FiTrash2, FiEdit2 } from 'react-icons/fi'
import { useChatStore } from '@/store/chat.store'
import { cn } from '@/lib/cn'

export function ConversationList() {
  const { conversations, activeId, create, setActive, remove, rename, toggleFavorite } = useChatStore()
  const [query, setQuery] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  const filtered = conversations.filter((c) => c.title.toLowerCase().includes(query.toLowerCase()))
  const pinned = filtered.filter((c) => c.pinned)
  const rest = filtered.filter((c) => !c.pinned)

  const commitRename = (id: string) => { if (draft.trim()) rename(id, draft.trim()); setEditing(null) }

  const Row = ({ c }: { c: (typeof conversations)[number] }) => (
    <div className={cn('group flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm transition-colors',
      activeId === c.id ? 'bg-blue/15 text-light' : 'text-dim hover:bg-white/5')}>
      <FiMessageSquare className="h-4 w-4 flex-shrink-0" />
      {editing === c.id ? (
        <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
          onBlur={() => commitRename(c.id)} onKeyDown={(e) => e.key === 'Enter' && commitRename(c.id)}
          className="min-w-0 flex-1 bg-transparent outline-none" />
      ) : (
        <button onClick={() => setActive(c.id)} className="min-w-0 flex-1 truncate text-left">{c.title}</button>
      )}
      <div className="flex flex-shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
        <button onClick={() => toggleFavorite(c.id)} aria-label="Favorite" className="rounded p-1 hover:text-gold">
          <FiStar className={cn('h-3 w-3', c.favorite && 'fill-gold text-gold')} />
        </button>
        <button onClick={() => { setEditing(c.id); setDraft(c.title) }} aria-label="Rename" className="rounded p-1 hover:text-light"><FiEdit2 className="h-3 w-3" /></button>
        <button onClick={() => remove(c.id)} aria-label="Delete" className="rounded p-1 hover:text-red-400"><FiTrash2 className="h-3 w-3" /></button>
      </div>
    </div>
  )

  return (
    <div className="flex h-full w-72 flex-shrink-0 flex-col border-r border-white/[0.06] bg-space/40">
      <div className="p-3">
        <button onClick={() => create()} className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-bright">
          <FiPlus className="h-4 w-4" /> New chat
        </button>
        <div className="mt-3 flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
          <FiSearch className="h-3.5 w-3.5 text-faint" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search chats"
            className="min-w-0 flex-1 bg-transparent text-xs text-light outline-none placeholder:text-faint" />
        </div>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
        {pinned.length > 0 && <div className="px-2 py-1 text-[10px] uppercase tracking-wide text-faint">Pinned</div>}
        {pinned.map((c) => <Row key={c.id} c={c} />)}
        {rest.length > 0 && pinned.length > 0 && <div className="px-2 py-1 text-[10px] uppercase tracking-wide text-faint">Recent</div>}
        {rest.map((c) => <Row key={c.id} c={c} />)}
        {conversations.length === 0 && <p className="px-3 py-6 text-center text-xs text-faint">No conversations yet.</p>}
      </div>
    </div>
  )
}
