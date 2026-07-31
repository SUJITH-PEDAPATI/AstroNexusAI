import { apiConfig } from '@/lib/config'

/**
 * Thrown when the backend explicitly rejects a request (4xx/5xx).
 * Callers catch this to surface a user-facing error instead of silently
 * falling back to demo mode.
 */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Reads the stored JWT (if any) from the persisted Zustand store without
 * importing the store directly — avoids a circular-dependency risk and works
 * both in client components and in plain service modules.
 */
function getToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem('anx-auth')
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return (parsed?.state?.token as string) ?? null
  } catch {
    return null
  }
}

/**
 * Core fetch wrapper used by every service.
 *
 * Behaviour:
 * - Attaches Authorization header automatically when a JWT is stored.
 * - Returns null ONLY on genuine network failure (backend unreachable) so
 *   callers can fall back to demo mode.
 * - Throws ApiError on 4xx/5xx so auth and validation errors surface to the UI.
 */
export async function http<T>(endpoint: string, init?: RequestInit): Promise<T | null> {
  const token = getToken()

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init?.headers as Record<string, string> | undefined),
  }
  if (token) headers['Authorization'] = `Bearer ${token}`

  try {
    const res = await fetch(`${apiConfig.baseUrl}${endpoint}`, {
      ...init,
      headers,
    })

    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const body = await res.json()
        message = body?.detail ?? body?.message ?? body?.error ?? message
      } catch { /* body not JSON */ }
      throw new ApiError(res.status, message)
    }

    return (await res.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    // Network-level failure → null → callers fall back to demo
    return null
  }
}

/**
 * Multipart upload variant — does NOT set Content-Type so the browser sets
 * the correct boundary for FormData automatically.
 */
export async function httpUpload<T>(endpoint: string, body: FormData): Promise<T | null> {
  const token = getToken()
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`

  try {
    const res = await fetch(`${apiConfig.baseUrl}${endpoint}`, {
      method: 'POST',
      headers,
      body,
    })
    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const b = await res.json()
        message = b?.detail ?? b?.message ?? b?.error ?? message
      } catch { /* body not JSON */ }
      throw new ApiError(res.status, message)
    }
    return (await res.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    return null
  }
}
