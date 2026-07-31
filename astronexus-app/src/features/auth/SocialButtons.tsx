'use client'
import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { FcGoogle } from 'react-icons/fc'
import { FiGithub, FiAlertCircle } from 'react-icons/fi'
import { useAuth } from '@/hooks/useAuth'
import { Spinner } from '@/components/ui/Spinner'

export function SocialButtons() {
  const { social } = useAuth()
  const router = useRouter()
  const params = useSearchParams()
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const go = async (provider: 'google' | 'github') => {
    setLoading(provider)
    setError(null)
    try {
      await social(provider)
      router.push(params.get('from') || '/dashboard')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Social sign-in failed. Please try again.')
    } finally {
      setLoading(null)
    }
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
