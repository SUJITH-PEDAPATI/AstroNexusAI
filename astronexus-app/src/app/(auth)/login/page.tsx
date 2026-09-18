'use client'
import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { motion } from 'framer-motion'
import { FiAlertCircle, FiEye, FiEyeOff, FiMail } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SocialButtons } from '@/features/auth/SocialButtons'
import { loginSchema, type LoginValues } from '@/features/auth/schemas'
import { useAuth, AuthError } from '@/hooks/useAuth'
import { EASE } from '@/lib/motion'

function LoginForm() {
  const { login, resendVerification, isDemo } = useAuth()
  const router = useRouter()
  const params = useSearchParams()

  const [submitting, setSubmitting]   = useState(false)
  const [authError,  setAuthError]    = useState<string | null>(
    params.get('error') ? decodeURIComponent(params.get('error')!) : null
  )
  const [showPw,     setShowPw]       = useState(false)
  const [needsVerify, setNeedsVerify] = useState(false)
  const [verifyEmail, setVerifyEmail] = useState('')
  const [resentMsg,   setResentMsg]   = useState<string | null>(null)

  const { register, handleSubmit, formState: { errors } } = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
  })

  const onSubmit = async (values: LoginValues) => {
    setSubmitting(true); setAuthError(null); setNeedsVerify(false); setResentMsg(null)
    try {
      await login(values.email, values.password)
      router.push(params.get('from') || '/dashboard')
    } catch (err: unknown) {
      if (err instanceof AuthError && err.code === 'email_not_confirmed') {
        setNeedsVerify(true)
        setVerifyEmail(values.email)
        setAuthError('Please verify your email before signing in.')
      } else {
        setAuthError(err instanceof Error ? err.message : 'Invalid email or password.')
      }
    } finally { setSubmitting(false) }
  }

  const handleResend = async () => {
    try {
      await resendVerification(verifyEmail)
      setResentMsg('Verification email resent. Check your inbox.')
    } catch { setResentMsg('Failed to resend. Try again later.') }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE.out }}
      className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl">

      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <Logo />
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Welcome back</h1>
          <p className="mt-1 text-sm text-dim">Sign in to your research workspace</p>
        </div>
      </div>

      {authError && (
        <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
          className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3" role="alert">
          <FiAlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <div className="text-sm text-red-300">
            <p>{authError}</p>
            {needsVerify && (
              <button onClick={handleResend} className="mt-1 text-xs underline hover:text-red-200">
                Resend verification email
              </button>
            )}
            {resentMsg && <p className="mt-1 text-xs text-silver-light">{resentMsg}</p>}
          </div>
        </motion.div>
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4" noValidate>
        <Input label="Email" type="email" placeholder="you@lab.edu" autoComplete="email"
          {...register('email')} error={errors.email?.message} />

        <div className="relative">
          <Input label="Password" type={showPw ? 'text' : 'password'} placeholder="••••••••"
            autoComplete="current-password" {...register('password')} error={errors.password?.message} />
          <button type="button" onClick={() => setShowPw(!showPw)}
            className="absolute right-3 top-9 text-faint hover:text-light" tabIndex={-1} aria-label="Toggle password">
            {showPw ? <FiEyeOff className="h-4 w-4" /> : <FiEye className="h-4 w-4" />}
          </button>
        </div>

        <div className="flex justify-end">
          <Link href="/forgot-password" className="text-xs text-faint hover:text-glow-soft">
            Forgot password?
          </Link>
        </div>

        <Button type="submit" size="lg" disabled={submitting} className="w-full">
          {submitting ? <Spinner size={16} /> : 'Sign in'}
        </Button>
      </form>

      <div className="my-6 flex items-center gap-3 text-xs text-faint">
        <div className="h-px flex-1 bg-white/10" /> or continue with <div className="h-px flex-1 bg-white/10" />
      </div>

      <SocialButtons />

      <p className="mt-6 text-center text-sm text-dim">
        No account?{' '}
        <Link href="/register" className="text-glow-soft hover:underline">Create one</Link>
      </p>

      {isDemo && (
        <p className="mt-3 text-center text-[11px] text-faint">
          Demo mode: any email + password signs you in when Supabase is not configured.
        </p>
      )}
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
