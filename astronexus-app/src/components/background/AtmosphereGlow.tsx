'use client'
/** CSS nebula + atmospheric glow layer echoing the video's steel-cyan limb. */
export function AtmosphereGlow() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <div className="absolute -bottom-1/3 left-1/4 h-[70vh] w-[70vh] -translate-x-1/2 rounded-full opacity-30 blur-[130px] animate-glow-pulse"
        style={{ background: 'radial-gradient(circle, rgba(108,162,193,0.4), transparent 65%)' }} />
      <div className="absolute -top-1/4 right-0 h-[50vh] w-[50vh] rounded-full opacity-15 blur-[140px]"
        style={{ background: 'radial-gradient(circle, rgba(232,225,211,0.35), transparent 60%)' }} />
      <div className="absolute left-1/2 top-1/2 h-[55vh] w-[85vh] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-10 blur-[150px]"
        style={{ background: 'radial-gradient(ellipse, rgba(11,61,145,0.4), transparent 70%)' }} />
    </div>
  )
}
