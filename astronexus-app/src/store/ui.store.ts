'use client'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UIState {
  sidebarOpen: boolean
  soundEnabled: boolean
  soundPrefSet: boolean
  aiGuideDismissed: boolean
  toggleSidebar: () => void
  setSidebar: (v: boolean) => void
  setSound: (v: boolean) => void
  dismissAiGuide: () => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      soundEnabled: false,
      soundPrefSet: false,
      aiGuideDismissed: false,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebar: (v) => set({ sidebarOpen: v }),
      setSound: (v) => set({ soundEnabled: v, soundPrefSet: true }),
      dismissAiGuide: () => set({ aiGuideDismissed: true }),
    }),
    { name: 'anx-ui' },
  ),
)
