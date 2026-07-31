'use client'
import { useEffect, type ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth.store'
import { Spinner } from '@/components/ui/Spinner'

/** Client-side guard that complements the edge middleware. Shows a graceful
 *  loading state during store rehydration, then redirects if unauthenticated. */
export function AuthGate({ children }: { children: ReactNode }) {
  const { user, hydrated } = useAuthStore()
  const router = useRouter()

  useEffect(() => {
    if (hydrated && !user) router.replace('/login')
  }, [hydrated, user, router])

  if (!hydrated || !user) {
    return (
      <div className="flex h-screen items-center justify-center bg-void">
        <Spinner size={28} />
      </div>
    )
  }
  return <>{children}</>
}
