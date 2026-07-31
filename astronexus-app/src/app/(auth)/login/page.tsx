'use client'
import { useState, Suspense } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { motion } from 'framer-motion'
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
  const [submitting, setSubmitting] = useState(false)
  const { register, handleSubmit, formState: { errors } } = useForm<LoginValues>({ resolver: zodResolver(loginSchema) })

  const onSubmit = async (values: LoginValues) => {
    setSubmitting(true)
    await login(values.email, values.password)
    router.push(params.get('from') || '/dashboard')
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE.out }}
      className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl">
      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <Logo />
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Welcome back</h1>
          <p className="mt-1 text-sm text-dim">Sign in to your research workspace</p>
        </div>
      </div>
      <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
        <Input label="Email" type="email" placeholder="you@lab.edu" {...register('email')} error={errors.email?.message} />
        <Input label="Password" type="password" placeholder="••••••••" {...register('password')} error={errors.password?.message} />
        <Button type="submit" size="lg" disabled={submitting} className="mt-2 w-full">
          {submitting ? <Spinner size={16} /> : 'Sign in'}
        </Button>
      </form>
      <div className="my-6 flex items-center gap-3 text-xs text-faint">
        <div className="h-px flex-1 bg-white/10" /> or continue with <div className="h-px flex-1 bg-white/10" />
      </div>
      <SocialButtons />
      <p className="mt-6 text-center text-sm text-dim">
        No account? <Link href="/register" className="text-glow-soft hover:underline">Create one</Link>
      </p>
      <p className="mt-3 text-center text-[11px] text-faint">Demo mode: any email + password signs you in.</p>
    </motion.div>
  )
}

export default function LoginPage() {
  return <Suspense fallback={<div className="flex justify-center"><Spinner size={28} /></div>}><LoginForm /></Suspense>
}
