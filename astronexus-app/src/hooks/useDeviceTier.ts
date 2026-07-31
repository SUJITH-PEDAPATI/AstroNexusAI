'use client'
import { useState, useEffect } from 'react'
import type { DeviceTier } from '@/types'
export function useDeviceTier(): DeviceTier {
  const [tier, setTier] = useState<DeviceTier>('high')
  useEffect(() => {
    const cores = navigator.hardwareConcurrency ?? 4
    const mem = (navigator as any).deviceMemory ?? 4
    const mobile = window.matchMedia('(max-width: 768px)').matches
    if (mobile || cores <= 4 || mem <= 2) setTier('low')
    else if (cores <= 6 || mem <= 4) setTier('mid')
    else setTier('high')
  }, [])
  return tier
}
