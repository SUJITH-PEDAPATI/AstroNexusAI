import type { User } from '@/types'
import { http } from './http'
import { apiConfig } from '@/lib/config'
import { uid } from '@/lib/utils'

/**
 * Auth service abstraction. Ships with a local demo implementation that works
 * with zero backend, so the app runs end-to-end out of the box. To use real
 * auth, implement these three methods against your API (or NextAuth/Clerk) —
 * the rest of the app only depends on this interface.
 */
const AVATAR_COLORS = ['#3B82F6', '#6CA2C1', '#0B3D91', '#D4B483', '#60A5FA']

export const authService = {
  async login(email: string, _password: string, provider: User['provider'] = 'email'): Promise<User> {
    const remote = await http<User>(apiConfig.endpoints.login, {
      method: 'POST',
      body: JSON.stringify({ email, password: _password }),
    })
    if (remote) return remote
    // Demo fallback
    return {
      id: uid(),
      name: email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) || 'Researcher',
      email,
      avatarColor: AVATAR_COLORS[email.length % AVATAR_COLORS.length],
      provider,
      createdAt: new Date().toISOString(),
    }
  },

  async register(name: string, email: string, _password: string): Promise<User> {
    const remote = await http<User>(apiConfig.endpoints.register, {
      method: 'POST',
      body: JSON.stringify({ name, email, password: _password }),
    })
    if (remote) return remote
    return {
      id: uid(),
      name,
      email,
      avatarColor: AVATAR_COLORS[name.length % AVATAR_COLORS.length],
      provider: 'email',
      createdAt: new Date().toISOString(),
    }
  },

  async social(provider: 'google' | 'github'): Promise<User> {
    // In production: redirect to OAuth. Demo: instant session.
    return this.login(`${provider}.user@astronexus.ai`, 'oauth', provider)
  },
}
