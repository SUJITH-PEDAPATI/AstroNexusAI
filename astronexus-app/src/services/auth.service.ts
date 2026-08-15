import type { User } from '@/types'
import { getSupabase, isSupabaseConfigured } from '@/lib/supabase'
import { uid } from '@/lib/utils'

const AVATAR_COLORS = ['#6E6E6E', '#8A8A8A', '#2C2C2C', '#A8A49E', '#9A9A9A']

function mapSupabaseUser(su: { id: string; email?: string; user_metadata?: Record<string, unknown> }): User {
  const email = su.email ?? ''
  const meta  = su.user_metadata ?? {}
  return {
    id:          su.id,
    name:        (meta.full_name as string) ?? (meta.name as string) ?? email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    email,
    avatarColor: AVATAR_COLORS[email.length % AVATAR_COLORS.length],
    avatarUrl:   (meta.avatar_url as string) ?? undefined,
    provider:    (meta.provider as User['provider']) ?? 'email',
    createdAt:   new Date().toISOString(),
  }
}

function demoUser(email: string, provider: User['provider'] = 'email', name?: string): User {
  return {
    id: uid(),
    name: name ?? (email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Researcher'),
    email,
    avatarColor: AVATAR_COLORS[email.length % AVATAR_COLORS.length],
    provider,
    createdAt: new Date().toISOString(),
  }
}

export class AuthError extends Error {
  constructor(message: string, public code?: string) {
    super(message)
    this.name = 'AuthError'
  }
}

export const authService = {
  /** Sign up with email + password. Sends verification email via Supabase. */
  async register(name: string, email: string, password: string): Promise<{ user: User; token: string; needsVerification: boolean }> {
    const sb = getSupabase()
    if (!sb) return { user: demoUser(email, 'email', name), token: '', needsVerification: false }

    const { data, error } = await sb.auth.signUp({
      email,
      password,
      options: { data: { full_name: name } },
    })
    if (error) throw new AuthError(error.message, error.code)
    if (!data.user) throw new AuthError('Registration failed. Please try again.')

    // Supabase returns session=null when email confirmation is required
    const needsVerification = !data.session
    return {
      user:  mapSupabaseUser(data.user),
      token: data.session?.access_token ?? '',
      needsVerification,
    }
  },

  /** Sign in with email + password. */
  async login(email: string, password: string): Promise<{ user: User; token: string }> {
    const sb = getSupabase()
    if (!sb) return { user: demoUser(email), token: '' }

    const { data, error } = await sb.auth.signInWithPassword({ email, password })
    if (error) {
      if (error.message.includes('Email not confirmed'))
        throw new AuthError('Please verify your email before signing in. Check your inbox.', 'email_not_confirmed')
      if (error.message.includes('Invalid login'))
        throw new AuthError('Invalid email or password.', 'invalid_credentials')
      throw new AuthError(error.message, error.code)
    }
    if (!data.user) throw new AuthError('Login failed.')
    return { user: mapSupabaseUser(data.user), token: data.session?.access_token ?? '' }
  },

  /** OAuth sign-in (Google / GitHub). Redirects to provider. */
  async social(provider: 'google' | 'github'): Promise<void> {
    const sb = getSupabase()
    if (!sb) return  // demo mode — handled in useAuth

    const { error } = await sb.auth.signInWithOAuth({
      provider,
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    })
    if (error) throw new AuthError(error.message)
  },

  /** Sign out. */
  async logout(): Promise<void> {
    const sb = getSupabase()
    if (sb) await sb.auth.signOut()
  },

  /** Send password reset email. */
  async resetPassword(email: string): Promise<void> {
    const sb = getSupabase()
    if (!sb) throw new AuthError('Password reset requires Supabase to be configured.')

    const { error } = await sb.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/reset-password`,
    })
    if (error) throw new AuthError(error.message)
  },

  /** Update password (used on the reset-password page). */
  async updatePassword(newPassword: string): Promise<void> {
    const sb = getSupabase()
    if (!sb) throw new AuthError('Password update requires Supabase.')

    const { error } = await sb.auth.updateUser({ password: newPassword })
    if (error) throw new AuthError(error.message)
  },

  /** Resend verification email. */
  async resendVerification(email: string): Promise<void> {
    const sb = getSupabase()
    if (!sb) return

    const { error } = await sb.auth.resend({ type: 'signup', email })
    if (error) throw new AuthError(error.message)
  },

  /** Get current session (for restore after refresh). */
  async getSession(): Promise<{ user: User; token: string } | null> {
    const sb = getSupabase()
    if (!sb) return null

    const { data: { session } } = await sb.auth.getSession()
    if (!session) return null
    return { user: mapSupabaseUser(session.user), token: session.access_token }
  },

  /** Listen for auth state changes (login, logout, token refresh, tab sync). */
  onAuthStateChange(callback: (event: string, user: User | null, token: string | null) => void) {
    const sb = getSupabase()
    if (!sb) return { unsubscribe: () => {} }

    const { data: { subscription } } = sb.auth.onAuthStateChange((event, session) => {
      const user  = session?.user ? mapSupabaseUser(session.user) : null
      const token = session?.access_token ?? null
      callback(event, user, token)
    })
    return { unsubscribe: () => subscription.unsubscribe() }
  },
}
