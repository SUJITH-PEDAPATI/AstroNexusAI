'use client'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'
import { setCookie, deleteCookie } from '@/lib/utils'

interface AuthState {
  user:      User | null
  token:     string | null
  hydrated:  boolean
  loading:   boolean
  setUser:   (user: User, token?: string) => void
  logout:    () => void
  setHydrated: () => void
  setLoading:  (loading: boolean) => void
}

const SESSION_COOKIE = 'anx_session'

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user:     null,
      token:    null,
      hydrated: false,
      loading:  true,

      setUser: (user, token) => {
        setCookie(SESSION_COOKIE, user.id)
        set({ user, token: token ?? null, loading: false })
      },

      logout: () => {
        deleteCookie(SESSION_COOKIE)
        set({ user: null, token: null, loading: false })
      },

      setHydrated: () => set({ hydrated: true }),
      setLoading:  (loading) => set({ loading }),
    }),
    {
      name: 'anx-auth',
      onRehydrateStorage: () => (state) => state?.setHydrated(),
    },
  ),
)
