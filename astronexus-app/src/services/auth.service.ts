import type { User } from '@/types'
import { http, ApiError } from './http'
import { apiConfig } from '@/lib/config'
import { uid } from '@/lib/utils'

const AVATAR_COLORS = ['#6E6E6E', '#8A8A8A', '#2C2C2C', '#A8A49E', '#9A9A9A']

function demoUser(email: string, provider: User['provider'] = 'email'): User {
  return {
    id: uid(),
    name:
      email
        .split('@')[0]
        .replace(/[._]/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase()) || 'Researcher',
    email,
    avatarColor: AVATAR_COLORS[email.length % AVATAR_COLORS.length],
    provider,
    createdAt: new Date().toISOString(),
  }
}

/**
 * Normalise the backend auth response.
 *
 * FastAPI jwt responses typically look like one of:
 *   { access_token, token_type, user: {...} }   ← preferred
 *   { access_token, token_type, ...userFields }  ← flat
 *   { token, user: {...} }                       ← some setups
 *
 * We handle all three so you don't have to change the backend.
 */
function parseAuthResponse(raw: Record<string, unknown>): { user: User; token: string } {
  const token =
    (raw.access_token as string) ??
    (raw.token as string) ??
    ''

  // Nested user object
  if (raw.user && typeof raw.user === 'object') {
    return { user: raw.user as User, token }
  }

  // Flat response — the user fields are at the top level alongside the token
  // Strip token fields; what remains are the user fields
  const rest = Object.fromEntries(
    Object.entries(raw).filter(([k]) => !['access_token','token_type','token'].includes(k))
  )
  return { user: rest as unknown as User, token }
}

export const authService = {
  /**
   * Returns { user, token } on success.
   * Throws ApiError on 4xx (wrong credentials).
   * Falls back to demo when backend is unreachable (null response).
   */
  async login(
    email: string,
    password: string,
    provider: User['provider'] = 'email',
  ): Promise<{ user: User; token: string }> {
    try {
      const raw = await http<Record<string, unknown>>(apiConfig.endpoints.login, {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      })
      if (raw) return parseAuthResponse(raw)
    } catch (err) {
      if (err instanceof ApiError) throw err
    }
    // Demo fallback — backend unreachable
    return { user: demoUser(email, provider), token: '' }
  },

  async register(
    name: string,
    email: string,
    password: string,
  ): Promise<{ user: User; token: string }> {
    try {
      const raw = await http<Record<string, unknown>>(apiConfig.endpoints.register, {
        method: 'POST',
        body: JSON.stringify({ name, email, password }),
      })
      if (raw) return parseAuthResponse(raw)
    } catch (err) {
      if (err instanceof ApiError) throw err
    }
    return {
      user: {
        id: uid(),
        name,
        email,
        avatarColor: AVATAR_COLORS[name.length % AVATAR_COLORS.length],
        provider: 'email',
        createdAt: new Date().toISOString(),
      },
      token: '',
    }
  },

  async social(provider: 'google' | 'github'): Promise<{ user: User; token: string }> {
    // In production, redirect to OAuth. Demo: instant session.
    return { user: demoUser(`${provider}.user@astronexus.ai`, provider), token: '' }
  },
}
