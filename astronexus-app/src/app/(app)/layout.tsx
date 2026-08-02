'use client'
import type { ReactNode } from 'react'
import { AuthGate } from '@/features/auth/AuthGate'
import { AppSidebar } from '@/components/layout/AppSidebar'
import { AppTopbar } from '@/components/layout/AppTopbar'
import { AIGuide } from '@/components/layout/AIGuide'
import { AtmosphereGlow } from '@/components/background/AtmosphereGlow'

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <div className="relative flex h-screen overflow-hidden">
        <AtmosphereGlow />
        <AppSidebar />
        <div className="relative z-10 flex min-w-0 flex-1 flex-col">
          <AppTopbar />
          <main className="flex-1 overflow-y-auto">{children}</main>
        </div>
        <AIGuide />
      </div>
    </AuthGate>
  )
}
