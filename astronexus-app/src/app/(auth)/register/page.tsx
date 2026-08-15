'use client'
import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { motion } from 'framer-motion'
import { FiAlertCircle, FiEye, FiEyeOff, FiMail } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SocialButtons } from '@/features/auth/SocialButtons'
import { registerSchema, type RegisterValues } from '@/features/auth/schemas'
import { useAuth, AuthError } from '@/hooks/useAuth'
import { EASE } from '@/lib/motion'

function RegisterForm() {
  const { register: doRegister, isDemo } = useAuth()
  const router = useRouter()
  const [submitting, setSubmitting] = useState(false)
  const [showPw, setShowPw]         = useState(false)
  const [error, setError]           = useState<string | null>(null)
  const [verifyScreen, setVerifyScreen] = useState(false)

  const { register, handleSubmit, formState: { errors } } = useForm<RegisterValues>({
    resolver: zodResolver(registerSchema),
  })

  const onSubmit = async (values: RegisterValues) => {
    setSubmitting(true); setError(null)
    try {
      const { needsVerification } = await doRegister(values.name, values.email, values.password)
      if (needsVerification) {
        setVerifyScreen(true)
      } else {
        router.push('/dashboard')
      }
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Registration failed.')
    } finally { setSubmitting(false) }
  }

  if (verifyScreen) {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: EASE.out }}
        className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl text-center">
        <div className="mb-6 flex flex-col items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white/[0.06]">
            <FiMail className="h-6 w-6 text-glow-soft" />
          </div>
          <h1 className="font-display text-2xl font-bold text-light">Check your email</h1>
          <p className="max-w-sm text-sm text-dim">
            We sent a verification link to your email. Click it to activate your account, then sign in.
          </p>
        </div>
        <Link href="/login" className="inline-block rounded-xl bg-white/[0.06] px-6 py-2.5 text-sm font-medium text-light hover:bg-white/[0.10]">
          Go to sign in
        </Link>
      </motion.div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE.out }}
      className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl">
      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <Logo />
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Create your account</h1>
          <p className="mt-1 text-sm text-dim">Start exploring the universe with AI</p>
        </div>
      </div>

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3" role="alert">
          <FiAlertCircle className="mt-0.5 h-4 w-4 text-red-400" />
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
        <Input label="Full name" placeholder="Ada Lovelace" {...register('name')} error={errors.name?.message} />
        <Input label="Email" type="email" placeholder="you@lab.edu" {...register('email')} error={errors.email?.message} />
        <div className="relative">
          <Input label="Password" type={showPw ? 'text' : 'password'} placeholder="••••••••"
            {...register('password')} error={errors.password?.message} />
          <button type="button" onClick={() => setShowPw(!showPw)}
            className="absolute right-3 top-9 text-faint hover:text-light" tabIndex={-1}>
            {showPw ? <FiEyeOff className="h-4 w-4" /> : <FiEye className="h-4 w-4" />}
          </button>
        </div>
        <Button type="submit" size="lg" disabled={submitting} className="mt-2 w-full">
          {submitting ? <Spinner size={16} /> : 'Create account'}
        </Button>
      </form>

      <div className="my-6 flex items-center gap-3 text-xs text-faint">
        <div className="h-px flex-1 bg-white/10" /> or sign up with <div className="h-px flex-1 bg-white/10" />
      </div>
      <SocialButtons />

      <p className="mt-6 text-center text-sm text-dim">
        Already have an account? <Link href="/login" className="text-glow-soft hover:underline">Sign in</Link>
      </p>

      {isDemo && (
        <p className="mt-3 text-center text-[11px] text-faint">
          Demo mode active — Supabase not configured.
        </p>
      )}
    </motion.div>
  )
}

export default function RegisterPage() {
  return <Suspense fallback={<div className="flex justify-center"><Spinner size={28} /></div>}><RegisterForm /></Suspense>
}
