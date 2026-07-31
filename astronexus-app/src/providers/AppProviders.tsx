'use client'
import { type ReactNode } from 'react'
import { QueryProvider } from './QueryProvider'

/** Root client providers wrapper. Smooth scroll is applied per-layout so app
 *  pages (with their own scroll containers) aren't hijacked. */
export function AppProviders({ children }: { children: ReactNode }) {
  return <QueryProvider>{children}</QueryProvider>
}
