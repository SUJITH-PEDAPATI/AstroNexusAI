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

      {/* About the video / what the platform does */}
      <section className="px-6 py-20">
        <div className="mx-auto max-w-3xl">
          <Reveal preset="fade">
            <span className="inline-flex items-center gap-2 rounded-full border border-glow/25 bg-glow/10 px-3.5 py-1.5 text-xs font-medium tracking-wide text-glow-soft backdrop-blur-sm">
              The view from orbit
            </span>
          </Reveal>
          <Reveal preset="blur" index={1}>
            <h2 className="mt-6 font-display text-3xl font-bold leading-[1.1] tracking-tight text-light md:text-4xl">
              One vantage point for the{' '}
              <span className="bg-gradient-to-r from-glow via-blue-bright to-gold bg-clip-text text-transparent">
                whole of space science.
              </span>
            </h2>
          </Reveal>

          <Reveal preset="fade" index={2}>
            <div className="mt-8 space-y-5 font-display text-[15px] font-light leading-8 tracking-[0.01em] text-dim md:text-base">
              <p className="first-letter:float-left first-letter:mr-3 first-letter:mt-1 first-letter:font-display first-letter:text-5xl first-letter:font-bold first-letter:leading-none first-letter:text-glow-soft">
                The film above is more than a backdrop. That lone observation satellite, drifting
                over a slowly turning Earth, is the metaphor at the heart of AstroNexusAI — a single
                point of view from which the sprawling landscape of space research finally becomes
                legible. Where a satellite gathers scattered signals and resolves them into a clear
                picture of the planet below, our platform gathers the scattered literature, imagery,
                and mission data of an entire field and resolves it into answers you can trust.
              </p>
              <p>
                At its core, AstroNexusAI is a research intelligence engine. Upload any scientific
                paper and it is parsed, split into section-aware passages, embedded with BGE-M3, and
                indexed into a Qdrant vector store. Ask a question and a hybrid retriever fuses dense
                semantic search with classical BM25 through reciprocal-rank fusion — then every
                answer is grounded to its exact source, cited down to the section and page, and scored
                for reliability before it ever reaches you. Nothing is invented; nothing is unverifiable.
              </p>
              <p>
                Around that engine lives a living knowledge graph. Every author, instrument, dataset,
                satellite, keyword, and institution becomes a node in a Neo4j graph that grows with
                each paper you ingest and each question you ask — turning a flat archive into a map you
                can traverse, and surfacing connections no single document could reveal on its own.
              </p>
              <p>
                Look outward and the same intelligence turns to Earth itself. Satellite imagery flows
                through DINOv2 features, SAM2 segmentation, and vision-language captioning to detect
                storms, map coastlines, trace floods, and describe what changed and why. Coordinating
                all of it, a multi-agent system — router, research, graph, satellite, and general
                agents orchestrated through LangGraph — decides which specialist should answer, and
                reaches for live scientific APIs like NASA, arXiv, and SIMBAD exactly when they are
                needed.
              </p>
              <p className="text-light/80">
                Research. Reason. Visualize. Discover. AstroNexusAI compresses a week of literature
                review into a single afternoon — so that discovery is limited only by imagination,
                never by the sheer volume of what there is to read.
              </p>
            </div>
          </Reveal>
        </div>
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
