import type { ReactNode } from 'react'

// Authentication removed — AuthGate is now a passthrough.
export function AuthGate({ children }: { children: ReactNode }) {
  return <>{children}</>
}
