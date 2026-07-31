'use client'
import { FiBookOpen, FiShare2, FiEye, FiZap } from 'react-icons/fi'
import { TbBrain, TbRoute } from 'react-icons/tb'
import { SectionHeading } from '@/components/ui/SectionHeading'
import { GlowCard } from '@/components/ui/GlowCard'
import { Reveal } from '@/components/animations/Reveal'
import { CountUp } from '@/components/ui/CountUp'

const CAPS = [
  { icon: FiBookOpen, title: 'Research QA', body: 'Ingest papers and ask grounded, cited questions.', accent: '#6CA2C1' },
  { icon: FiShare2, title: 'Knowledge Graph', body: 'Neo4j links every entity across your corpus.', accent: '#3B82F6' },
  { icon: TbBrain, title: 'Reasoning Core', body: 'Chain-of-thought drafting refined and grounded.', accent: '#7DD3FC' },
  { icon: FiEye, title: 'Satellite Vision', body: 'Detect, segment, and caption Earth imagery.', accent: '#D4B483' },
  { icon: TbRoute, title: 'Multi-Agent', body: 'Five specialists coordinated via LangGraph.', accent: '#60A5FA' },
  { icon: FiZap, title: 'Live APIs', body: 'NASA, arXiv, SIMBAD selected automatically.', accent: '#A5D8F0' },
]

const STATS = [
  { v: 0.94, d: 2, s: '', label: 'Hit Rate' },
  { v: 0.76, d: 2, s: '', label: 'MRR' },
  { v: 5, d: 0, s: '', label: 'AI Agents' },
  { v: 9, d: 0, s: '+', label: 'Live APIs' },
]

/** The home story: capabilities + proof, told with Apple-style reveals. */
export function StorySection() {
  return (
    <div className="relative bg-gradient-to-b from-transparent via-void/60 to-void">
      <section className="px-6 py-32">
        <div className="mx-auto max-w-6xl">
          <SectionHeading badge="The Platform" title="One system for the whole of" highlight="space science."
            subtitle="Everything a research team needs to read, connect, and reason across the literature — and the sky." />
          <div className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {CAPS.map((c, i) => {
              const Icon = c.icon
              return (
                <Reveal key={c.title} preset="fade" index={i}>
                  <GlowCard accent={c.accent} className="h-full">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl" style={{ background: `${c.accent}1a`, border: `1px solid ${c.accent}33` }}>
                      <Icon className="h-5 w-5" style={{ color: c.accent }} />
                    </div>
                    <h3 className="mt-5 font-display text-lg font-semibold text-light">{c.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-dim">{c.body}</p>
                  </GlowCard>
                </Reveal>
              )
            })}
          </div>
        </div>
      </section>

      <section className="px-6 pb-32">
        <div className="mx-auto max-w-5xl">
          <Reveal preset="scale">
            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-3xl border border-white/[0.07] bg-white/[0.02] md:grid-cols-4">
              {STATS.map((s) => (
                <div key={s.label} className="bg-white/[0.015] px-6 py-10 text-center">
                  <div className="font-display text-4xl font-bold text-glow-soft"><CountUp target={s.v} decimals={s.d} suffix={s.s} /></div>
                  <div className="mt-2 text-sm text-dim">{s.label}</div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>
    </div>
  )
}
