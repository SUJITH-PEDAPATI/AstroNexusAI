'use client'
import { motion } from 'framer-motion'
import { FiVolume2, FiVolumeX } from 'react-icons/fi'
import { useSpeech } from '@/hooks/useSpeech'

interface Props { text: string; msgId: string }

export function ReadAloudButton({ text, msgId }: Props) {
  const { speak, isSpeaking, supported } = useSpeech()
  if (!supported || !text.trim()) return null

  const active = isSpeaking(msgId)

  return (
    <motion.button
      onClick={() => speak(text, msgId)}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      title={active ? 'Stop reading' : 'Read aloud'}
      aria-label={active ? 'Stop reading' : 'Read aloud'}
      className="group mt-2 flex items-center gap-1 rounded-lg px-1.5 py-1 text-faint transition-colors hover:bg-white/[0.05] hover:text-silver"
    >
      <motion.span
        animate={active ? { scale: [1, 1.15, 1] } : { scale: 1 }}
        transition={active ? { repeat: Infinity, duration: 1.4, ease: 'easeInOut' } : {}}
      >
        {active
          ? <FiVolumeX className="h-3.5 w-3.5 text-silver-light" />
          : <FiVolume2 className="h-3.5 w-3.5" />}
      </motion.span>
      <span className="text-[10px] opacity-0 transition-opacity group-hover:opacity-100">
        {active ? 'Stop' : 'Read aloud'}
      </span>
    </motion.button>
  )
}
