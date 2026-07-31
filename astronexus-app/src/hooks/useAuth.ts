'use client'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth.store'
import { authService } from '@/services/auth.service'
import type { User } from '@/types'

/** Ergonomic auth facade over the store + service. */
export function useAuth() {
  const { user, hydrated, setUser, logout } = useAuthStore()
  const router = useRouter()

  return {
    user,
    hydrated,
    isAuthenticated: !!user,
    async login(email: string, password: string) {
      const u = await authService.login(email, password)
      setUser(u)
      return u
    },
    async register(name: string, email: string, password: string) {
      const u = await authService.register(name, email, password)
      setUser(u)
      return u
    },
    async social(provider: 'google' | 'github') {
      const u = await authService.social(provider)
      setUser(u)
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
