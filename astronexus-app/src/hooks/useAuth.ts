'use client'
import { useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth.store'
import { authService, AuthError } from '@/services/auth.service'
import { isSupabaseConfigured } from '@/lib/supabase'
import type { User } from '@/types'

export { AuthError }

export function useAuth() {
  const { user, token, hydrated, loading, setUser, logout: clearStore, setLoading } = useAuthStore()
  const router    = useRouter()
  const initRef   = useRef(false)

  // Restore session on mount + listen for auth state changes (tab sync, token refresh)
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    // Restore session from Supabase
    authService.getSession().then(session => {
      if (session) {
        setUser(session.user, session.token)
      } else {
        setLoading(false)
      }
    }).catch(() => setLoading(false))

    // Listen for auth events (SIGNED_IN, SIGNED_OUT, TOKEN_REFRESHED, etc.)
    const { unsubscribe } = authService.onAuthStateChange((event, u, t) => {
      if (event === 'SIGNED_OUT') {
        clearStore()
      } else if (u) {
        setUser(u, t ?? undefined)
      }
    })

    return () => unsubscribe()
  }, [setUser, clearStore, setLoading])

  return {
    user,
    token,
    hydrated,
    loading,
    isAuthenticated: !!user,
    isDemo: !isSupabaseConfigured(),

    async login(email: string, password: string): Promise<User> {
      setLoading(true)
      try {
        const { user: u, token: t } = await authService.login(email, password)
        setUser(u, t)
        return u
      } finally {
        setLoading(false)
      }
    },

    async register(name: string, email: string, password: string): Promise<{ user: User; needsVerification: boolean }> {
      setLoading(true)
      try {
        const { user: u, token: t, needsVerification } = await authService.register(name, email, password)
        if (!needsVerification) setUser(u, t)
        return { user: u, needsVerification }
      } finally {
        setLoading(false)
      }
    },

    async social(provider: 'google' | 'github'): Promise<void> {
      if (!isSupabaseConfigured()) {
        // Demo mode — instant login
        const { uid } = await import('@/lib/utils')
        setUser({
          id: uid(), name: `${provider} User`, email: `${provider}@demo.astronexus.ai`,
          avatarColor: '#8A8A8A', provider, createdAt: new Date().toISOString(),
        })
        return
      }
      await authService.social(provider)
      // Redirect happens via Supabase — no further action needed
    },

    async resetPassword(email: string): Promise<void> {
      await authService.resetPassword(email)
    },

    async updatePassword(newPassword: string): Promise<void> {
      await authService.updatePassword(newPassword)
    },

    async resendVerification(email: string): Promise<void> {
      await authService.resendVerification(email)
    },

    updateUser(patch: Partial<User>) {
      if (user) setUser({ ...user, ...patch })
    },

    signOut() {
      authService.logout().catch(() => {})
      clearStore()
      router.push('/login')
    },
  }
}
