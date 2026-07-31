'use client'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth.store'
import { authService } from '@/services/auth.service'
import type { User } from '@/types'

export function useAuth() {
  const { user, token, hydrated, setUser, logout } = useAuthStore()
  const router = useRouter()

  return {
    user,
    token,
    hydrated,
    isAuthenticated: !!user,

    async login(email: string, password: string): Promise<User> {
      // Throws ApiError on bad credentials — login page catches it.
      const { user: u, token: t } = await authService.login(email, password)
      setUser(u, t)
      return u
    },

    async register(name: string, email: string, password: string): Promise<User> {
      const { user: u, token: t } = await authService.register(name, email, password)
      setUser(u, t)
      return u
    },

    async social(provider: 'google' | 'github'): Promise<User> {
      const { user: u, token: t } = await authService.social(provider)
      setUser(u, t)
      return u
    },

    updateUser(patch: Partial<User>) {
      if (user) setUser({ ...user, ...patch })
    },

    signOut() {
      logout()
      router.push('/login')
    },
  }
}
