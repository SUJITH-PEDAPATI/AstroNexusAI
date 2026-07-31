'use client'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'
import { setCookie, deleteCookie } from '@/lib/utils'

interface AuthState {
  user: User | null
  hydrated: boolean
  setUser: (user: User) => void
  logout: () => void
  setHydrated: () => void
}

const SESSION_COOKIE = 'anx_session'

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      hydrated: false,
      setUser: (user) => {
        setCookie(SESSION_COOKIE, user.id)
        set({ user })
      },
      logout: () => {
        deleteCookie(SESSION_COOKIE)
        set({ user: null })
      },
      setHydrated: () => set({ hydrated: true }),
    }),
    {
      name: 'anx-auth',
      onRehydrateStorage: () => (state) => state?.setHydrated(),
    },
  ),
)
