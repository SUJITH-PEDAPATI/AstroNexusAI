'use client'
import { motion } from 'framer-motion'
import { FiFileText } from 'react-icons/fi'
import { TbSparkles } from 'react-icons/tb'
import type { ChatMessage } from '@/types'
import { Avatar } from '@/components/ui/Avatar'
import { Markdown } from './Markdown'

export function MessageBubble({ msg, userName, userColor, streaming }: {
  msg: ChatMessage; userName: string; userColor: string; streaming?: boolean
}) {
  const isUser = msg.role === 'user'
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {isUser ? <Avatar name={userName} color={userColor} size={32} />
        : <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-glow/30 bg-glow/15 text-glow-soft"><TbSparkles className="h-4 w-4" /></span>}
      <div className={`max-w-[78%] rounded-2xl px-4 py-3 ${isUser ? 'rounded-tr-sm bg-blue text-white' : 'rounded-tl-sm border border-white/[0.08] bg-white/[0.03]'}`}>
        {isUser ? <p className="text-sm leading-relaxed">{msg.content}</p> : <Markdown content={msg.content + (streaming ? ' ▍' : '')} />}
        {!isUser && msg.grade && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-white/[0.06] pt-3">
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-bold text-emerald-400">Grade {msg.grade}</span>
            {msg.citations?.map((c, i) => (
              <span key={i} className="inline-flex items-center gap-1 rounded-lg border border-white/[0.06] bg-white/[0.03] px-2 py-0.5 font-mono text-[10px] text-dim">
                <FiFileText className="h-2.5 w-2.5" />{c.section} p.{c.page} · {c.score.toFixed(2)}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}
