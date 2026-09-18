import type { Citation } from '@/types'
import { http } from './http'
import { apiConfig } from '@/lib/config'

export interface WebSource { title: string; url: string; source: string; snippet: string }
export interface ChatResult {
  answer:      string
  grade:       string
  citations:   Citation[]
  search_type:  string       // local_rag | web_search | hybrid
  search_label: string       // '📄 Research Papers' | '🌐 Web Search' | '📄 + 🌐 Hybrid'
  web_sources:  WebSource[]
}

// ── Demo fallbacks — only when the backend is genuinely unreachable ───────────

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
  answer: '⚠️ **Unable to reach the AstroNexus AI backend right now.**\n\nThis is usually a temporary network issue. Please try again in a moment.\n\nIf the problem persists, make sure the backend server is running on `http://localhost:8000`.',
  grade:        'N/A',
  citations:    [],
  search_type:  'local_rag',
  search_label: '📄 Research Papers',
  web_sources:  [],
}

// ── Backend response shape ────────────────────────────────────────────────────
interface BackendChatResponse {
  answer:          string
  grade:           string
  citations:       { section: string; page: number; score: number }[]
  confidence?:     string
  grounding_score?: number
  query_type?:     string
  is_reliable?:    boolean
  search_type?:    string
  search_label?:   string
  web_sources?:    { title: string; url: string; source: string; snippet: string }[]
}

export const chatService = {
  /**
   * POST /chat  →  { answer, grade, citations }
   *
   * Uses the shared http() client so the Authorization header is attached
   * automatically whenever the user is logged in.
   *
   * Falls back to demo answers ONLY on network failure (TypeError).
   * Real backend errors (4xx / 5xx) are thrown so the chat UI can display them.
   */
  async resolve(query: string, paperId?: string): Promise<ChatResult> {
    try {
      const data = await http<BackendChatResponse>(apiConfig.endpoints.chat, {
        method: 'POST',
        body: JSON.stringify({
          query,
          paper_id:             paperId ?? null,
          conversation_history: null,   // future: pass from chat store
        }),
      })

      if (data) {
        return {
          answer:    data.answer,
          grade:     data.grade ?? 'B',
          citations:    (data.citations ?? []).map((c) => ({
            section: c.section ?? 'Unknown',
            page:    c.page    ?? 0,
            score:   c.score   ?? 0,
          })),
          search_type:  data.search_type  ?? 'local_rag',
          search_label: data.search_label ?? '📄 Research Papers',
          web_sources:  data.web_sources  ?? [],
        }
      }

      // http() returned null → network down → demo
      const hit = DEMO_ANSWERS.find((d) => d.match.test(query))
      return hit
          ? { answer: hit.answer, grade: 'A', citations: hit.citations, search_type: 'local_rag', search_label: '📄 Research Papers', web_sources: [] }
          : FALLBACK

    } catch (err) {
      // http() throws ApiError on 4xx/5xx — propagate so the UI shows the message
      throw err
    }
  },

  /** Token-by-token streaming effect applied to the answer string. */
  async *stream(text: string): AsyncGenerator<string> {
    const tokens = text.split(/(\s+)/)
    for (const tok of tokens) {
      await new Promise((r) => setTimeout(r, 16 + Math.random() * 34))
      yield tok
    }
  },
}
