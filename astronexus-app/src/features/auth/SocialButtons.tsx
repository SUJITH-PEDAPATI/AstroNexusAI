'use client'
import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { FcGoogle } from 'react-icons/fc'
import { FiGithub } from 'react-icons/fi'
import { useAuth } from '@/hooks/useAuth'
import { Spinner } from '@/components/ui/Spinner'

export function SocialButtons() {
  const { social } = useAuth()
  const router = useRouter()
  const params = useSearchParams()
  const [loading, setLoading] = useState<string | null>(null)

  const go = async (provider: 'google' | 'github') => {
    setLoading(provider)
    await social(provider)
    router.push(params.get('from') || '/dashboard')
  }

  return (
    <div className="grid grid-cols-2 gap-3">
      <button onClick={() => go('google')} disabled={!!loading}
        className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] py-3 text-sm text-light transition-colors hover:border-white/20 disabled:opacity-50">
        {loading === 'google' ? <Spinner size={16} /> : <FcGoogle className="h-4 w-4" />} Google
      </button>
      <button onClick={() => go('github')} disabled={!!loading}
        className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] py-3 text-sm text-light transition-colors hover:border-white/20 disabled:opacity-50">
        {loading === 'github' ? <Spinner size={16} /> : <FiGithub className="h-4 w-4" />} GitHub
      </button>
    </div>
  )
}
