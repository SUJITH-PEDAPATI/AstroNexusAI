import type { Citation } from '@/types'
import { apiConfig } from '@/lib/config'

export interface ChatResult { answer: string; grade: string; citations: Citation[] }

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
  answer:
    "AstroNexusAI routes your question to the right agent, retrieves grounded evidence, and returns a cited answer with a reliability grade. Connect a backend via `NEXT_PUBLIC_API_URL` to query your own ingested corpus.",
  grade: 'B',
  citations: [],
}

export const chatService = {
  /** Resolve an answer — real backend when available, else a rich local demo. */
  async resolve(query: string, paperId?: string): Promise<ChatResult> {
    try {
      const res = await fetch(`${apiConfig.baseUrl}${apiConfig.endpoints.chat}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, paper_id: paperId }),
      })
      if (res.ok) return (await res.json()) as ChatResult
    } catch {
      /* fall through to demo */
    }
    const hit = DEMO_ANSWERS.find((d) => d.match.test(query))
    return hit
      ? { answer: hit.answer, grade: 'A', citations: hit.citations }
      : FALLBACK
  },

  /** Async generator that yields tokens for a typing/streaming effect. */
  async *stream(text: string): AsyncGenerator<string> {
    const tokens = text.split(/(\s+)/)
    for (const tok of tokens) {
      await new Promise((r) => setTimeout(r, 16 + Math.random() * 34))
      yield tok
    }
  },
}
