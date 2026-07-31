import type { ReactNode } from 'react'
import { AtmosphereGlow } from '@/components/background/AtmosphereGlow'

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-4">
      <AtmosphereGlow />
      {/* subtle poster backdrop */}
      <div className="pointer-events-none absolute inset-0 opacity-30">
        <img src="/images/earth-still.jpg" alt="" className="h-full w-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-b from-void/80 via-void/60 to-void" />
      </div>
      <div className="relative z-10 w-full max-w-md">{children}</div>
    </div>
  )
}
