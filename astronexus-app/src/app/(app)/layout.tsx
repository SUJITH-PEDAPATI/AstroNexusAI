'use client'
import dynamic from 'next/dynamic'
import type { ReactNode } from 'react'
import { AppSidebar } from '@/components/layout/AppSidebar'
import { AppTopbar } from '@/components/layout/AppTopbar'
import { AIGuide } from '@/components/layout/AIGuide'
import { AtmosphereGlow } from '@/components/background/AtmosphereGlow'

const SpaceCanvas = dynamic(
  () => import('@/components/background/SpaceCanvas').then(m => m.SpaceCanvas),
  { ssr: false },
)

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex h-screen overflow-hidden">
      <SpaceCanvas />
      <AtmosphereGlow />
      <AppSidebar />
      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <AppTopbar />
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
      <AIGuide />
    </div>
  )
}
