'use client'
import { SectionHeading } from '@/components/ui/SectionHeading'
import { Reveal } from '@/components/animations/Reveal'
import { Parallax } from '@/components/animations/Parallax'
import { SplitText } from '@/components/animations/SplitText'

const VALUES = [
  { title: 'Grounded', body: 'Every answer is traced to a source. No hallucinated citations, ever.' },
  { title: 'Transparent', body: 'You can see which agent answered and why — reasoning is never a black box.' },
  { title: 'Fast', body: 'Seconds from question to cited answer, across an entire corpus.' },
]

export default function AboutPage() {
  return (
    <div className="relative">
      {/* Hero */}
      <section className="flex min-h-[80vh] items-center px-6 pt-24">
        <div className="mx-auto max-w-4xl text-center">
          <h1 className="font-display text-5xl font-bold leading-[1.05] tracking-tight text-light md:text-7xl">
            <SplitText text="Built for the people who" />
            <br />
            <span className="bg-gradient-to-r from-glow via-blue-bright to-gold bg-clip-text text-transparent"><SplitText text="push the frontier." delay={0.15} /></span>
          </h1>
          <Reveal preset="fade" index={3}>
            <p className="mx-auto mt-8 max-w-2xl text-lg leading-relaxed text-dim">
              AstroNexusAI began as a research internship at NIT Kurukshetra with one question:
              what if no scientist ever had to lose a discovery to the sheer volume of the literature?
            </p>
          </Reveal>
        </div>
      </section>

      {/* Still image band */}
      <section className="px-6 py-16">
        <Parallax speed={0.12} className="mx-auto max-w-5xl">
          <div className="relative overflow-hidden rounded-3xl border border-white/[0.08]">
            <img src="/images/satellite-still.jpg" alt="Satellite over Earth" className="h-[420px] w-full object-cover" />
            <div className="absolute inset-0 bg-gradient-to-t from-void via-transparent to-transparent" />
          </div>
        </Parallax>
      </section>

      {/* Values */}
      <section className="px-6 py-24">
        <div className="mx-auto max-w-6xl">
          <SectionHeading badge="Principles" title="What we" highlight="believe." />
          <div className="mt-16 grid gap-5 md:grid-cols-3">
            {VALUES.map((v, i) => (
              <Reveal key={v.title} preset="fade" index={i}>
                <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-8">
                  <div className="font-display text-2xl font-bold text-glow-soft">{v.title}</div>
                  <p className="mt-3 text-sm leading-relaxed text-dim">{v.body}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
