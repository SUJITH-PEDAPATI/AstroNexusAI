'use client'
import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { FiCheck } from 'react-icons/fi'
import { Card } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { Avatar } from '@/components/ui/Avatar'

const schema = z.object({ name: z.string().min(2), email: z.string().email() })
type Values = z.infer<typeof schema>

const ACTIVITY = [
  { label: 'Conversations', value: 128 },
  { label: 'Papers ingested', value: 34 },
  { label: 'Queries run', value: 1204 },
]

export default function ProfilePage() {
  const [saved, setSaved] = useState(false)
  const { register, handleSubmit, formState: { errors, isDirty } } = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: { name: user?.name ?? '', email: user?.email ?? '' },
  })

  const onSubmit = (values: Values) => {
    updateUser(values)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="mx-auto max-w-3xl p-6 md:p-8">
      <h1 className="mb-8 font-display text-2xl font-bold text-light">Profile</h1>

      <Card className="mb-6 flex items-center gap-5 p-6">
        {user && <Avatar name={user.name} color={user.avatarColor} size={64} />}
        <div>
          <div className="font-display text-lg font-bold text-light">{user?.name}</div>
          <div className="text-sm text-dim">{user?.email}</div>
          <div className="mt-1 text-xs capitalize text-faint">Signed in with {user?.provider}</div>
        </div>
      </Card>

      <div className="mb-6 grid grid-cols-3 gap-4">
        {ACTIVITY.map((a) => (
          <Card key={a.label} className="p-5 text-center">
            <div className="font-display text-2xl font-bold text-glow-soft">{a.value.toLocaleString()}</div>
            <div className="mt-1 text-xs text-dim">{a.label}</div>
          </Card>
        ))}
      </div>

      <Card className="p-6">
        <h2 className="mb-4 font-display text-sm font-semibold text-light">Edit details</h2>
        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
          <Input label="Full name" {...register('name')} error={errors.name?.message} />
          <Input label="Email" type="email" {...register('email')} error={errors.email?.message} />
          <Button type="submit" disabled={!isDirty} className="self-start">
            {saved ? <><FiCheck className="h-4 w-4" /> Saved</> : 'Save changes'}
          </Button>
        </form>
      </Card>
    </div>
  )
}
