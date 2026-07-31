'use client'
import { useState, useRef, useEffect, useCallback } from 'react'
import { FiSend, FiPaperclip, FiDownload, FiX } from 'react-icons/fi'
import { TbSparkles } from 'react-icons/tb'
import { ConversationList } from '@/features/chat/ConversationList'
import { MessageBubble } from '@/features/chat/MessageBubble'
import { useChatStore } from '@/store/chat.store'
import { useAuth } from '@/hooks/useAuth'
import { chatService } from '@/services/chat.service'
import { uid } from '@/lib/utils'

const SUGGESTIONS = [
  'What is the Origins Space Telescope?',
  'Explain infrared spectroscopy with equations',
  'How does SAR detect floods?',
]

export default function ChatPage() {
  const { user } = useAuth()
  const { conversations, activeId, create, setActive, addMessage, updateLastAssistant } = useChatStore()
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [files, setFiles] = useState<string[]>([])
  const endRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const active = conversations.find((c) => c.id === activeId) ?? null

  useEffect(() => {
    if (!activeId && conversations.length > 0) setActive(conversations[0].id)
  }, [activeId, conversations, setActive])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [active?.messages.length, busy])

  const send = useCallback(async (text: string) => {
    if (!text.trim() || busy) return
    let id = activeId
    if (!id) id = create()
    setInput('')
    setFiles([])
    addMessage(id, { id: uid(), role: 'user', content: text, createdAt: Date.now() })
    setBusy(true)

    const result = await chatService.resolve(text)
    addMessage(id, { id: uid(), role: 'assistant', content: '', createdAt: Date.now() })
    let acc = ''
    for await (const tok of chatService.stream(result.answer)) {
      acc += tok
      updateLastAssistant(id, { content: acc })
    }
    updateLastAssistant(id, { content: acc, grade: result.grade, citations: result.citations })
    setBusy(false)
  }, [activeId, busy, create, addMessage, updateLastAssistant])

  const exportChat = () => {
    if (!active) return
    const md = active.messages.map((m) => `**${m.role === 'user' ? 'You' : 'AstroNexus'}:**\n\n${m.content}`).join('\n\n---\n\n')
    const blob = new Blob([`# ${active.title}\n\n${md}`], { type: 'text/markdown' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${active.title.slice(0, 30)}.md`
    a.click()
  }

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <div className="hidden md:block"><ConversationList /></div>

      <div className="flex min-w-0 flex-1 flex-col">
        {active && (
          <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-3">
            <h2 className="truncate font-display text-sm font-semibold text-light">{active.title}</h2>
            <button onClick={exportChat} className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-dim transition-colors hover:text-light" aria-label="Export chat">
              <FiDownload className="h-3.5 w-3.5" /> Export
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-6 py-6">
          {!active || active.messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center gap-6 text-center">
              <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-glow/25 bg-glow/10 text-glow-soft"><TbSparkles className="h-7 w-7" /></span>
              <div>
                <h1 className="font-display text-2xl font-bold text-light">How can I help with your research?</h1>
                <p className="mt-2 text-sm text-dim">Ask a scientific question, or upload a paper to ground the conversation.</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)} className="rounded-full border border-white/[0.08] bg-white/[0.03] px-4 py-2 text-xs text-dim transition-colors hover:border-glow/30 hover:text-light">{s}</button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6">
              {active.messages.map((m, i) => (
                <MessageBubble key={m.id} msg={m} userName={user?.name ?? 'You'} userColor={user?.avatarColor ?? '#3B82F6'}
                  streaming={busy && i === active.messages.length - 1 && m.role === 'assistant' && !m.grade} />
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-white/[0.06] px-6 py-4">
          <div className="mx-auto max-w-3xl">
            {files.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-2">
                {files.map((f) => (
                  <span key={f} className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.03] px-2.5 py-1 text-xs text-dim">
                    <FiPaperclip className="h-3 w-3" />{f}
                    <button onClick={() => setFiles((p) => p.filter((x) => x !== f))} aria-label="Remove"><FiX className="h-3 w-3 hover:text-red-400" /></button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-end gap-2 rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-2 focus-within:border-glow/40">
              <button onClick={() => fileRef.current?.click()} className="p-2 text-dim hover:text-light" aria-label="Attach file"><FiPaperclip className="h-4 w-4" /></button>
              <input ref={fileRef} type="file" accept=".pdf" className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) setFiles((p) => [...p, f.name]) }} />
              <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={1} placeholder="Ask anything about space science…"
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
                className="max-h-32 flex-1 resize-none bg-transparent py-2 text-sm text-light outline-none placeholder:text-faint" />
              <button onClick={() => send(input)} disabled={!input.trim() || busy}
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue text-white transition-colors hover:bg-blue-bright disabled:opacity-40" aria-label="Send">
                <FiSend className="h-4 w-4" />
              </button>
            </div>
            <p className="mt-2 text-center text-[10px] text-faint">AstroNexus can make mistakes. Answers are grounded to your ingested papers.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
