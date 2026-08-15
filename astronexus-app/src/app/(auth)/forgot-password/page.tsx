'use client'
import { useState } from 'react'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { motion } from 'framer-motion'
import { FiMail, FiCheck, FiAlertCircle } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { useAuth, AuthError } from '@/hooks/useAuth'
import { EASE } from '@/lib/motion'

const schema = z.object({ email: z.string().email('Enter a valid email') })
type Values = z.infer<typeof schema>

export default function ForgotPasswordPage() {
  const { resetPassword } = useAuth()
  const [submitting, setSubmitting] = useState(false)
  const [sent, setSent]   = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { register, handleSubmit, formState: { errors } } = useForm<Values>({ resolver: zodResolver(schema) })

  const onSubmit = async (values: Values) => {
    setSubmitting(true); setError(null)
    try {
      await resetPassword(values.email)
      setSent(true)
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Failed to send reset email.')
    } finally { setSubmitting(false) }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE.out }}
      className="rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl">
      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <Logo />
        <div>
          <h1 className="font-display text-2xl font-bold text-light">Reset password</h1>
          <p className="mt-1 text-sm text-dim">Enter your email to receive a reset link</p>
        </div>
      </div>

      {sent ? (
        <div className="flex flex-col items-center gap-3 py-4 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/[0.06]">
            <FiCheck className="h-5 w-5 text-silver-light" />
          </div>
          <p className="text-sm text-light">Reset link sent!</p>
          <p className="text-xs text-dim">Check your inbox for the password reset email.</p>
          <Link href="/login" className="mt-2 text-sm text-glow-soft hover:underline">Back to sign in</Link>
        </div>
      ) : (
        <>
          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3" role="alert">
              <FiAlertCircle className="mt-0.5 h-4 w-4 text-red-400" />
              <p className="text-sm text-red-300">{error}</p>
            </motion.div>
          )}
          <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
            <Input label="Email" type="email" placeholder="you@lab.edu" autoComplete="email" {...register('email')} error={errors.email?.message} />
            <Button type="submit" size="lg" disabled={submitting} className="mt-2 w-full">
              {submitting ? <Spinner size={16} /> : 'Send reset link'}
            </Button>
          </form>
          <p className="mt-6 text-center text-sm text-dim">
            <Link href="/login" className="text-glow-soft hover:underline">Back to sign in</Link>
          </p>
        </>
      )}
    </motion.div>
  )
}
