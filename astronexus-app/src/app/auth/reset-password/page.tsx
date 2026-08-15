'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { motion } from 'framer-motion'
import { FiCheck, FiAlertCircle } from 'react-icons/fi'
import { Logo } from '@/components/ui/Logo'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { useAuth, AuthError } from '@/hooks/useAuth'
import { EASE } from '@/lib/motion'

const schema = z.object({
  password:        z.string().min(6, 'At least 6 characters'),
  confirmPassword: z.string().min(6, 'At least 6 characters'),
}).refine(d => d.password === d.confirmPassword, {
  message: 'Passwords do not match', path: ['confirmPassword'],
})
type Values = z.infer<typeof schema>

export default function ResetPasswordPage() {
  const { updatePassword } = useAuth()
  const router = useRouter()
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone]   = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { register, handleSubmit, formState: { errors } } = useForm<Values>({ resolver: zodResolver(schema) })

  const onSubmit = async (values: Values) => {
    setSubmitting(true); setError(null)
    try {
      await updatePassword(values.password)
      setDone(true)
      setTimeout(() => router.push('/login'), 2000)
    } catch (err) {
      setError(err instanceof AuthError ? err.message : 'Failed to update password.')
    } finally { setSubmitting(false) }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: EASE.out }}
        className="w-full max-w-md rounded-3xl border border-white/[0.08] bg-space/70 p-8 backdrop-blur-2xl">
        <div className="mb-8 flex flex-col items-center gap-4 text-center">
          <Logo />
          <h1 className="font-display text-2xl font-bold text-light">Set new password</h1>
        </div>
        {done ? (
          <div className="flex flex-col items-center gap-3 py-4 text-center">
            <FiCheck className="h-6 w-6 text-silver-light" />
            <p className="text-sm text-light">Password updated! Redirecting…</p>
          </div>
        ) : (
          <>
            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="mb-4 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3">
                <FiAlertCircle className="mt-0.5 h-4 w-4 text-red-400" />
                <p className="text-sm text-red-300">{error}</p>
              </motion.div>
            )}
            <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
              <Input label="New password" type="password" placeholder="••••••••" {...register('password')} error={errors.password?.message} />
              <Input label="Confirm password" type="password" placeholder="••••••••" {...register('confirmPassword')} error={errors.confirmPassword?.message} />
              <Button type="submit" size="lg" disabled={submitting} className="mt-2 w-full">
                {submitting ? <Spinner size={16} /> : 'Update password'}
              </Button>
            </form>
          </>
        )}
      </motion.div>
    </div>
  )
}
