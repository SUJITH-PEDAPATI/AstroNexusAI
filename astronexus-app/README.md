# AstroNexusAI — Space Intelligence Platform (Frontend)

A complete, production-ready **SaaS application** frontend for **AstroNexusAI** — an AI-powered space research platform unifying RAG, knowledge graphs, satellite vision, and multi-agent reasoning.

The entire interface is built around the uploaded cinematic satellite video (with its original soundtrack) and inherits its exact visual language: **deep space black, NASA blue, steel-cyan atmospheric glow, metallic gray, and soft warm-white** — a moody mission-control identity.

Built with **Next.js 15 (App Router) · React 18 · TypeScript · TailwindCSS · GSAP · Framer Motion · Three.js / React Three Fiber / Drei · Lenis · React Query · Zustand · React Hook Form · Zod**.

---

## ✨ What's inside

A full multi-page application — not a landing page:

| Route | Description |
| --- | --- |
| `/` | Immersive fullscreen video hero + cinematic Apple-style story scroll, with a smart **mute/unmute soundtrack** that remembers your preference |
| `/about` | Cinematic story page with parallax |
| `/login`, `/register` | Auth with React Hook Form + Zod, email + Google + GitHub |
| `/dashboard` | Stat cards, area/bar/donut charts, recent activity, bookmarks |
| `/chat` | ChatGPT-quality chat — streaming, Markdown, **LaTeX**, code highlighting, citations, file upload, rename, delete, **export to Markdown** |
| `/chat/history` | Timeline with search, filters, favorites, pinning, restore, delete |
| `/chat/results` | Saved graded answers with citations |
| `/research` | PDF upload with a live processing pipeline (parse → chunk → embed → index → graph) |
| `/papers` | Ingested corpus grid with status badges |
| `/knowledge-graph` | **Interactive 3D Three.js graph** — drag to rotate, scroll to zoom, click to expand |
| `/vision-ai` | Satellite image upload with Detect / Segment / Explain modes |
| `/profile` | Editable profile + activity stats |
| `/settings` | Preference toggles (sound, sidebar, AI guide) + accessibility notes |

Plus a persistent **floating navbar**, collapsible **app sidebar** with breadcrumbs, and a dismissible **floating AI guide** that adapts its tips to the current page.

---

## 🚀 Quick start

```bash
npm install       # clean install — no --legacy-peer-deps needed
npm run dev       # http://localhost:3000
npm run build     # production build (verified: 0 errors, 0 warnings)
npm run start     # serve the production build
npm run lint
npm run format
```

> **Node 18.18+ / 20+** recommended. The stack uses stable React 18 + `@react-three/fiber@8`, so installation and builds are clean and reproducible.

**Try it immediately:** the app runs fully standalone with a demo backend fallback. Sign in on `/login` with **any email + password** to explore every page.

---

## 🎬 The hero video & soundtrack

The uploaded clip ships optimized in `public/videos/` (H.264 MP4 with faststart + VP9 WebM, both retaining the original audio) plus a poster.

The home hero implements the full audio spec:
- Autoplays **muted** (always allowed by browsers), poster shown until ready — no flicker.
- An elegant control **fades** the soundtrack in/out (no hard cuts) instead of restarting.
- Your choice is **remembered** (persisted in the UI store); on return, sound re-enables on your first interaction to satisfy autoplay policies.

To swap the clip, replace `public/videos/hero.mp4` / `.webm` and `public/images/hero-poster.jpg`.

---

## 📁 Architecture (feature-based, `src/`)

```
src/
├── app/                      # Next.js App Router
│   ├── (marketing)/          # / (home), /about  — public, smooth-scrolled
│   ├── (auth)/               # /login, /register — centered auth shell
│   └── (app)/                # authenticated shell: sidebar + topbar + AI guide
│       ├── dashboard/ chat/ (history, results) research/ papers/
│       ├── knowledge-graph/ vision-ai/ profile/ settings/
│   ├── layout.tsx            # root: fonts, KaTeX/highlight CSS, providers
│   └── middleware.ts         # route protection via session cookie
├── features/                 # feature modules (auth, home, chat, dashboard,
│                             #   knowledge-graph, vision) — colocated logic + UI
├── components/               # shared: ui/, layout/, background/, animations/, common/
├── services/                 # http client, auth.service, chat.service (API abstraction)
├── store/                    # Zustand: auth, chat, ui (all persisted)
├── hooks/                    # useAuth, useDeviceTier, useReducedMotion, useMediaQuery
├── providers/                # QueryProvider, SmoothScrollProvider, AppProviders
├── lib/                      # cn, utils, motion, config, navigation
├── types/  utils/  styles/
└── public/                   # videos, images, textures, models, icons
```

---

## 🔌 Backend integration

Everything is decoupled from transport through **`src/services/`** and configured via **`src/lib/config.ts`** — no hardcoded endpoints.

1. Copy `.env.example` → `.env.local` and set:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
   ```
2. Endpoints live in `apiConfig.endpoints` (`/chat`, `/ingest`, `/conversations`, `/papers`, `/graph`, `/vision`, …). Point them at your FastAPI routes.
3. Every service method calls the real API first and **falls back to a local demo** if it's unreachable — so the UI never breaks and development needs no backend. Remove the fallbacks when you go live.

Ready to wire: **Auth**, **Chat**, **History**, **Dashboard**, **RAG (ingest)**, **Neo4j (graph)**, **Qdrant**, and **Vision** — each already has a typed seam in `services/` or the relevant feature module.

### Authentication

Auth is implemented as a clean, swappable abstraction: a `useAuth` hook over a persisted Zustand store, a session cookie read by the edge **middleware** to protect the `(app)` routes, and an `authService` with email + Google + GitHub entry points. It works out of the box in demo mode.

To use **NextAuth/Auth.js or Clerk**, implement the three methods in `src/services/auth.service.ts` (and swap the cookie check in `src/middleware.ts` for the provider's middleware). Nothing else in the app needs to change — every component depends only on `useAuth`. The `.env.example` already includes the OAuth variable placeholders.

---

## ⚡ Performance & accessibility

- **Static-first**: all 14 routes prerender to static HTML. Heavy 3D (`knowledge-graph`) and the markdown/LaTeX/highlight stack (`chat`) are **dynamically imported** (`ssr: false`) so they never bloat other pages.
- **Tiered 3D**: `useDeviceTier` scales starfield density and DPR to the device.
- **GPU-friendly** transforms/opacity, passive listeners, additive-blended points, refs over state in hot paths.
- **Reduced motion**: disables Lenis, the camera drift, and all animations, with instant reveals — respected app-wide.
- **Keyboard nav, ARIA labels, focus-visible rings, and WCAG-minded contrast** throughout.
- `optimizePackageImports` for `react-icons`, `framer-motion`, `drei`, and `date-fns`.

---

## 🎨 Design tokens (from the video)

| Token | Hex | Role |
| --- | --- | --- |
| `void` / `space` | `#07080A` / `#0A0E14` | Deep space black |
| `nasa` | `#0B3D91` | NASA blue (deep) |
| `blue` | `#3B82F6` | Interactive blue |
| `glow` | `#6CA2C1` | Steel-cyan atmospheric glow — primary accent |
| `glow-soft` | `#7DD3FC` | Highlights |
| `steel` | `#6C6F72` | Metallic gray |
| `gold` | `#E8E1D3` | Soft warm-white / satellite foil |

All tokens live in `tailwind.config.ts`; motion easings/variants in `src/lib/motion.ts`. Rebrand by editing those two files.

---

## 🛠 Tech stack

Next.js 15.1 · React 18.3 · TypeScript 5.7 · TailwindCSS 3.4 · GSAP 3.12 · Framer Motion 11 · Three.js 0.171 · @react-three/fiber 8 · @react-three/drei 9 · Lenis 1.1 · @tanstack/react-query 5 · Zustand 5 · React Hook Form 7 · Zod 3 · react-markdown + remark-math + rehype-katex + rehype-highlight · date-fns · react-icons.

---

Built for the AstroNexusAI research platform · NIT Kurukshetra.
