'use client'
import dynamic from 'next/dynamic'
import { SmoothScrollProvider } from '@/providers/SmoothScrollProvider'
import { MarketingNav } from '@/components/layout/MarketingNav'
import { Footer } from '@/components/layout/Footer'
import { AtmosphereGlow } from '@/components/background/AtmosphereGlow'
import { MouseFollower } from '@/components/common/MouseFollower'

const SpaceCanvas = dynamic(
  () => import('@/components/background/SpaceCanvas').then(m => m.SpaceCanvas),
  { ssr: false },
)

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <SmoothScrollProvider>
      <SpaceCanvas />
      <AtmosphereGlow />
      <MouseFollower />
      <MarketingNav />
      <main className="relative z-10">{children}</main>
      <Footer />
    </SmoothScrollProvider>
  )
}
