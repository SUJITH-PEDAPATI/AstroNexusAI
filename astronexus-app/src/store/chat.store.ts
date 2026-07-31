'use client'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Conversation, ChatMessage } from '@/types'
import { uid } from '@/lib/utils'

interface ChatState {
  conversations: Conversation[]
  activeId: string | null
  create: () => string
  setActive: (id: string) => void
  addMessage: (id: string, msg: ChatMessage) => void
  updateLastAssistant: (id: string, patch: Partial<ChatMessage>) => void
  rename: (id: string, title: string) => void
  remove: (id: string) => void
  togglePin: (id: string) => void
  toggleFavorite: (id: string) => void
  setFolder: (id: string, folder: string) => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      conversations: [],
      activeId: null,
      create: () => {
        const id = uid()
        const now = Date.now()
        const convo: Conversation = {
          id, title: 'New conversation', messages: [],
          pinned: false, favorite: false, createdAt: now, updatedAt: now,
        }
        set((s) => ({ conversations: [convo, ...s.conversations], activeId: id }))
        return id
      },
      setActive: (id) => set({ activeId: id }),
      addMessage: (id, msg) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id
              ? {
                  ...c,
                  messages: [...c.messages, msg],
                  updatedAt: Date.now(),
                  title:
                    c.messages.length === 0 && msg.role === 'user'
                      ? msg.content.slice(0, 48)
                      : c.title,
                }
              : c,
          ),
        })),
      updateLastAssistant: (id, patch) =>
        set((s) => ({
          conversations: s.conversations.map((c) => {
            if (c.id !== id) return c
            const msgs = [...c.messages]
            for (let i = msgs.length - 1; i >= 0; i--) {
              if (msgs[i].role === 'assistant') { msgs[i] = { ...msgs[i], ...patch }; break }
            }
            return { ...c, messages: msgs, updatedAt: Date.now() }
          }),
        })),
      rename: (id, title) =>
        set((s) => ({ conversations: s.conversations.map((c) => (c.id === id ? { ...c, title } : c)) })),
      remove: (id) =>
        set((s) => {
          const conversations = s.conversations.filter((c) => c.id !== id)
          return { conversations, activeId: s.activeId === id ? (conversations[0]?.id ?? null) : s.activeId }
        }),
      togglePin: (id) =>
        set((s) => ({ conversations: s.conversations.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c)) })),
      toggleFavorite: (id) =>
        set((s) => ({ conversations: s.conversations.map((c) => (c.id === id ? { ...c, favorite: !c.favorite } : c)) })),
      setFolder: (id, folder) =>
        set((s) => ({ conversations: s.conversations.map((c) => (c.id === id ? { ...c, folder } : c)) })),
    }),
    { name: 'anx-chat' },
  ),
)
