'use client'
import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { motion } from 'framer-motion'
import { FiAlertCircle } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SocialButtons } from '@/features/auth/SocialButtons'
import { loginSchema, type LoginValues } from '@/features/auth/schemas'
import { useAuth } from '@/hooks/useAuth'
import { EASE } from '@/lib/motion'

function LoginForm() {
  const { login } = useAuth()
  const router = useRouter()
  const params = useSearchParams()

  // Separate loading and credential-error state so the UI is precise.
  const [submitting, setSubmitting] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginValues>({ resolver: zodResolver(loginSchema) })

  const onSubmit = async (values: LoginValues) => {
    setSubmitting(true)
    setAuthError(null) // clear any previous error

    try {
      await login(values.email, values.password)
      // Success — redirect to the page the user originally tried to reach,
      // or fall back to the dashboard.
      router.push(params.get('from') || '/dashboard')
    } catch (err: unknown) {
      // The backend returned a 4xx / 5xx (ApiError) — show the message.
      const message =
        err instanceof Error ? err.message : 'Invalid email or password. Please try again.'
      setAuthError(message)
    } finally {
      // Always re-enable the button, whether login succeeded or failed.
      setSubmitting(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE.out }}
      className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl"
    >
      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <Logo />
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Welcome back</h1>
          <p className="mt-1 text-sm text-dim">Sign in to your research workspace</p>
        </div>
      </div>

      {/* Credential error banner — only appears on a rejected login */}
      {authError && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3"
          role="alert"
        >
          <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{authError}</p>
        </motion.div>
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4" noValidate>
        <Input
          label="Email"
          type="email"
          placeholder="you@lab.edu"
          autoComplete="email"
          aria-invalid={!!errors.email}
          {...register('email')}
          error={errors.email?.message}
        />
        <Input
          label="Password"
          type="password"
          placeholder="••••••••"
          autoComplete="current-password"
          aria-invalid={!!errors.password}
          {...register('password')}
          error={errors.password?.message}
        />
        <Button type="submit" size="lg" disabled={submitting} className="mt-2 w-full">
          {submitting ? <Spinner size={16} /> : 'Sign in'}
        </Button>
      </form>

      <div className="my-6 flex items-center gap-3 text-xs text-faint">
        <div className="h-px flex-1 bg-white/10" /> or continue with{' '}
        <div className="h-px flex-1 bg-white/10" />
      </div>

      <SocialButtons />

      <p className="mt-6 text-center text-sm text-dim">
        No account?{' '}
        <Link href="/register" className="text-glow-soft hover:underline">
          Create one
        </Link>
      </p>

      <p className="mt-3 text-center text-[11px] text-faint">
        Demo mode: any email + password signs you in when the backend is offline.
      </p>
    </motion.div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="flex justify-center"><Spinner size={28} /></div>}>
      <LoginForm />
    </Suspense>
  )
}
