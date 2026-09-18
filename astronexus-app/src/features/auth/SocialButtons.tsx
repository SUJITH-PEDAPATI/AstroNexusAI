'use client'
import { useState } from 'react'
import { FcGoogle } from 'react-icons/fc'
import { FiGithub, FiAlertCircle } from 'react-icons/fi'
import { useAuth } from '@/hooks/useAuth'
import { Spinner } from '@/components/ui/Spinner'

export function SocialButtons() {
  const { social, isDemo } = useAuth()
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError]     = useState<string | null>(null)

  const go = async (provider: 'google' | 'github') => {
    setLoading(provider)
    setError(null)
    try {
      await social(provider)
      // For Supabase OAuth: social() triggers a full-page redirect to the provider.
      // Do NOT call router.push() here — it races with the redirect and causes a flash.
      // The redirect chain is: Provider → /auth/callback → /dashboard.
      //
      // For demo mode: social() sets the user in the store synchronously, so
      // AuthGate will handle the redirect via its effect. No router.push needed.
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Social sign-in failed.')
      setLoading(null)
    }
    // NOTE: setLoading(null) is NOT called on success because the page is
    // navigating away. Calling it would briefly flash the buttons before redirect.
  }

  return (
    <div className="flex flex-col gap-3">
      {error && (
        <div className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2.5" role="alert">
          <FiAlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-red-400" />
          <p className="text-xs text-red-300">{error}</p>
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={() => go('google')}
          disabled={!!loading}
          className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] py-3 text-sm text-light transition-colors hover:border-white/20 disabled:opacity-50"
        >
          {loading === 'google' ? <Spinner size={16} /> : <FcGoogle className="h-4 w-4" />} Google
        </button>
        <button
          onClick={() => go('github')}
          disabled={!!loading}
          className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] py-3 text-sm text-light transition-colors hover:border-white/20 disabled:opacity-50"
        >
          {loading === 'github' ? <Spinner size={16} /> : <FiGithub className="h-4 w-4" />} GitHub
        </button>
      </div>
    </div>
  )
}
