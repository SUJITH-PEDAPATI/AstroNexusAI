import type { Citation } from '@/types'
import { apiConfig } from '@/lib/config'

export interface ChatResult { answer: string; grade: string; citations: Citation[] }

// ── Demo fallbacks — only used when the backend is genuinely unreachable ──────

const DEMO_ANSWERS: { match: RegExp; answer: string; citations: Citation[] }[] = [
  {
    match: /origins|ost|telescope/i,
    answer:
      'The **Origins Space Telescope (OST)** is a proposed far-infrared observatory covering **5–600 μm**. Its three science goals are:\n\n1. Tracing the *rise of metals* and galaxy evolution across cosmic time\n2. Mapping the growth of planetary systems and water trails\n3. Searching for **biosignatures** in exoplanet atmospheres\n\nIt uses a 5.9 m mirror actively cooled to **4 K**, enabling sensitivity limited only by the astronomical background.',
    citations: [
      { section: 'Introduction', page: 1, score: 0.71 },
      { section: 'Science Goals', page: 3, score: 0.67 },
    ],
  },
  {
    match: /spectroscopy|infrared/i,
    answer:
      'Infrared spectroscopy measures how molecules absorb IR light at characteristic wavelengths. For exoplanets, transit spectroscopy captures starlight filtered through the atmosphere, revealing absorption features from $H_2O$, $CO_2$, and $CH_4$.\n\nThe key relation is the Beer–Lambert law:\n\n$$ I = I_0 \\, e^{-\\tau(\\lambda)} $$\n\nwhere $\\tau(\\lambda)$ is the optical depth at wavelength $\\lambda$.',
    citations: [{ section: 'Methods', page: 5, score: 0.63 }],
  },
  {
    match: /sar|flood|radar/i,
    answer:
      'Synthetic Aperture Radar (SAR) detects floods by measuring backscatter. Calm floodwater acts as a **specular reflector**, bouncing the radar pulse away from the sensor, so flooded pixels appear **dark** in the returned image. Change-detection between pre- and post-event acquisitions isolates newly inundated areas.',
    citations: [{ section: 'Remote Sensing', page: 8, score: 0.66 }],
  },
]

const FALLBACK: ChatResult = {
  answer: 'Backend is unreachable. Start the FastAPI server and set NEXT_PUBLIC_API_URL.',
  grade: 'N/A',
  citations: [],
}

// ── Backend response shape from POST /research ────────────────────────────────
interface ResearchResponse {
  answer:          string
  confidence:      string   // "HIGH" | "MEDIUM" | "LOW"
  grounding_score: number
  relevance_score: number
  citations:       { section?: string; page?: number; score?: number }[]
  warnings:        string[]
  is_reliable:     boolean
  sources:         string
  ollama_used:     boolean
  gemini_used:     boolean
}

/** Map backend confidence string → single-letter grade the UI already uses. */
function confidenceToGrade(conf: string): string {
  if (conf === 'HIGH')   return 'A'
  if (conf === 'MEDIUM') return 'B'
  return 'C'
}

/** Read the stored JWT without importing the Zustand store (avoids circular dep). */
function getToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem('anx-auth')
    if (!raw) return null
    return (JSON.parse(raw)?.state?.token as string) ?? null
  } catch {
    return null
  }
}

export const chatService = {
  /**
   * Send a query to the real backend POST /research.
   *
   * Key details:
   *  - Uses JSON — the backend expects a structured object.
   *  - Maps the ResearchResponse to the ChatResult shape the UI expects.
   *  - Falls back to demo answers only on genuine network failure (fetch throws),
   *    NOT on API errors — those bubble up as thrown errors.
   */
  async resolve(query: string, paperId?: string): Promise<ChatResult> {
    const token = getToken()
    // The new backend endpoint expects a JSON payload matching ChatRequest
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (token) headers['Authorization'] = `Bearer ${token}`

    const body = JSON.stringify({
      query,
      paper_id: paperId || null,
      conversation_history: [],
    })

    try {
      const res = await fetch(
        `${apiConfig.baseUrl}/chat`,   // ← correct backend endpoint
        { method: 'POST', headers, body },
      )

      if (res.ok) {
        // The backend returns ChatResponse with { answer, grade, citations, ... }
        const data = await res.json()
        const citations: Citation[] = (data.citations ?? []).map((c: any) => ({
          section: c.section ?? 'Unknown',
          page: c.page ?? 0,
          score: c.score ?? 0,
        }))
        return {
          answer: data.answer ?? 'No answer provided.',
          grade: data.grade ?? 'N/A',
          citations,
        }
      }

      // Backend returned a non-200 status — surface the error text
      let errText = `Backend error ${res.status}`
      try { errText = ((await res.json()) as { detail?: string }).detail ?? errText }
      catch { /* body not JSON */ }
      throw new Error(errText)

    } catch (err) {
      // Re-throw real errors so the chat UI can display them instead of showing a demo answer
      if (err instanceof TypeError) {
        // TypeError = fetch failed = network down → fall back to demo
        const hit = DEMO_ANSWERS.find((d) => d.match.test(query))
        return hit ? { answer: hit.answer, grade: 'A', citations: hit.citations } : FALLBACK
      }
      // Anything else (HTTP error, JSON parse error) → propagate
      throw err
    }
  },

  /** Token-by-token streaming effect applied to the answer string from the backend. */
  async *stream(text: string): AsyncGenerator<string> {
    const tokens = text.split(/(\s+)/)
    for (const tok of tokens) {
      await new Promise((r) => setTimeout(r, 16 + Math.random() * 34))
      yield tok
    }
  },
}
