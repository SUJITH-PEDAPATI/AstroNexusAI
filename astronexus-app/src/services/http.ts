import { apiConfig } from '@/lib/config'

/**
 * Thin fetch wrapper. Every request funnels through here so auth headers,
 * base URL, and error handling live in one place. Returns null on failure so
 * callers can fall back to local demo behaviour when the backend is offline.
 */
export async function http<T>(endpoint: string, init?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(`${apiConfig.baseUrl}${endpoint}`, {
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      ...init,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return (await res.json()) as T
  } catch {
    return null
  }
}
