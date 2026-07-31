'use client'
import { Card } from '@/components/ui/Card'
import { useUIStore } from '@/store/ui.store'
import { cn } from '@/lib/cn'

function Toggle({ on, onClick, label, desc }: { on: boolean; onClick: () => void; label: string; desc: string }) {
  return (
    <div className="flex items-center justify-between py-4">
      <div>
        <div className="text-sm font-medium text-light">{label}</div>
        <div className="text-xs text-dim">{desc}</div>
      </div>
      <button onClick={onClick} role="switch" aria-checked={on} aria-label={label}
        className={cn('relative h-6 w-11 flex-shrink-0 rounded-full transition-colors', on ? 'bg-blue' : 'bg-white/10')}>
        <span className={cn('absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform', on ? 'left-0.5 translate-x-5' : 'left-0.5')} />
      </button>
    </div>
  )
}

export default function SettingsPage() {
  const { soundEnabled, setSound, aiGuideDismissed, dismissAiGuide, sidebarOpen, setSidebar } = useUIStore()

  return (
    <div className="mx-auto max-w-2xl p-6 md:p-8">
      <h1 className="mb-8 font-display text-2xl font-bold text-light">Settings</h1>

      <Card className="mb-6 px-6">
        <h2 className="border-b border-white/[0.06] py-4 font-display text-sm font-semibold text-light">Preferences</h2>
        <div className="divide-y divide-white/[0.04]">
          <Toggle on={soundEnabled} onClick={() => setSound(!soundEnabled)} label="Hero soundtrack" desc="Play the original audio on the home video by default" />
          <Toggle on={sidebarOpen} onClick={() => setSidebar(!sidebarOpen)} label="Expanded sidebar" desc="Keep the navigation sidebar expanded" />
          <Toggle on={!aiGuideDismissed} onClick={dismissAiGuide} label="AI guide" desc="Show the floating assistant with contextual tips" />
        </div>
      </Card>

      <Card className="px-6">
        <h2 className="border-b border-white/[0.06] py-4 font-display text-sm font-semibold text-light">Accessibility</h2>
        <div className="py-4">
          <p className="text-sm text-dim">
            AstroNexusAI respects your system <span className="text-light">reduced-motion</span> preference automatically —
            animations, smooth scrolling, and the 3D background all soften or disable when it&apos;s enabled. Adjust this in your OS display settings.
          </p>
        </div>
      </Card>

      <p className="mt-6 text-center text-xs text-faint">Preferences are saved locally to your browser.</p>
    </div>
  )
}
