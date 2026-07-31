'use client'
import { useState } from 'react'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { FiX, FiHelpCircle } from 'react-icons/fi'
import { TbSparkles } from 'react-icons/tb'
import { useUIStore } from '@/store/ui.store'
import { EASE } from '@/lib/motion'

const TIPS: Record<string, string> = {
  '/dashboard': 'This is your mission control. Track usage, recent research, and jump back into any conversation.',
  '/chat': 'Ask any scientific question. Upload a paper first to get grounded, cited answers.',
  '/chat/history': 'Every conversation is saved here. Pin favorites, organize into folders, or search across them.',
  '/research': 'Drop a PDF to ingest it — chunking, embedding, and graph extraction happen automatically.',
  '/papers': 'Your ingested corpus lives here. Each paper is ready to query from Chat.',
  '/knowledge-graph': 'Explore how entities connect. Drag to rotate, scroll to zoom, click a node to expand.',
  '/vision-ai': 'Upload satellite imagery to detect, segment, and caption features with vision models.',
  '/profile': 'Manage your identity and see your activity across the platform.',
  '/settings': 'Tune your experience — motion, sound, and theme preferences.',
}

/** Dismissible bottom-right AI guide that adapts its tip to the current page. */
export function AIGuide() {
  const pathname = usePathname()
  const { aiGuideDismissed, dismissAiGuide } = useUIStore()
  const [open, setOpen] = useState(false)

  const tipKey = Object.keys(TIPS).find((k) => pathname === k || pathname.startsWith(k + '/')) ?? '/dashboard'
  const tip = TIPS[tipKey]

  if (aiGuideDismissed && !open) {
    return (
      <button onClick={() => setOpen(true)}
        className="fixed bottom-5 right-5 z-40 flex h-12 w-12 items-center justify-center rounded-full border border-glow/30 bg-space/80 text-glow-soft shadow-lg backdrop-blur-xl transition-transform hover:scale-105"
        aria-label="Open AI guide">
        <FiHelpCircle className="h-5 w-5" />
      </button>
    )
  }

  return (
    <AnimatePresence>
      {(open || !aiGuideDismissed) && (
        <motion.div initial={{ opacity: 0, y: 20, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }} transition={{ duration: 0.4, ease: EASE.out }}
          className="fixed bottom-5 right-5 z-40 w-[300px] overflow-hidden rounded-2xl border border-white/10 bg-space/90 backdrop-blur-2xl">
          <div className="flex items-center gap-2 border-b border-white/[0.06] px-4 py-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-glow/15 text-glow-soft">
              <TbSparkles className="h-4 w-4" />
            </span>
            <span className="flex-1 text-sm font-semibold text-light">AstroNexus Guide</span>
            <button onClick={() => { setOpen(false); dismissAiGuide() }} className="rounded-md p-1 text-dim hover:text-light" aria-label="Dismiss">
              <FiX className="h-4 w-4" />
            </button>
          </div>
          <div className="p-4">
            <p className="text-sm leading-relaxed text-dim">{tip}</p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
