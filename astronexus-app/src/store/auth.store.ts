'use client'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'
import { setCookie, deleteCookie } from '@/lib/utils'

interface AuthState {
  user: User | null
  token: string | null   // ← added: JWT from backend
  hydrated: boolean
  setUser: (user: User, token?: string) => void
  logout: () => void
  setHydrated: () => void
}

const SESSION_COOKIE = 'anx_session'

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      hydrated: false,
      setUser: (user, token) => {
        setCookie(SESSION_COOKIE, user.id)
        set({ user, token: token ?? null })
      },
      logout: () => {
        deleteCookie(SESSION_COOKIE)
        set({ user: null, token: null })
      },
      setHydrated: () => set({ hydrated: true }),
    }),
    {
      name: 'anx-auth',
      onRehydrateStorage: () => (state) => state?.setHydrated(),
    },
  ),
)
