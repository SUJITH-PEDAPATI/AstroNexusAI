'use client'
import { motion } from 'framer-motion'
import { FiFileText, FiExternalLink } from 'react-icons/fi'
import { TbSparkles } from 'react-icons/tb'
import type { ChatMessage } from '@/types'
import type { WebSource } from '@/services/chat.service'
import { Avatar } from '@/components/ui/Avatar'
import { Markdown } from './Markdown'

export function MessageBubble({ msg, userName, userColor, streaming, searchLabel, webSources }: {
  msg: ChatMessage; userName: string; userColor: string; streaming?: boolean
  searchLabel?: string; webSources?: WebSource[]
}) {
  const isUser = msg.role === 'user'
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {isUser ? <Avatar name={userName} color={userColor} size={32} />
        : <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-white/[0.12] bg-white/[0.06] text-silver"><TbSparkles className="h-4 w-4" /></span>}
      <div className={`max-w-[78%] rounded-[16px] px-4 py-3 ${isUser ? 'rounded-tr-sm border border-white/[0.10] bg-[#2E2E2E] text-light' : 'rounded-tl-sm border border-white/[0.06] bg-white/[0.03] backdrop-blur-xl'}`}>
        {isUser ? <p className="text-sm leading-relaxed">{msg.content}</p> : <Markdown content={msg.content + (streaming ? ' ▍' : '')} />}
        {!isUser && msg.grade && (
          <div className="mt-3 flex flex-col gap-2 border-t border-white/[0.06] pt-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="rounded-full border border-white/[0.10] bg-white/[0.06] px-2 py-0.5 text-[10px] font-bold text-silver-light">Grade {msg.grade}</span>
              {searchLabel && (
                <span className="rounded-full border border-white/10 bg-white/[0.03] px-2 py-0.5 text-[10px] text-dim">{searchLabel}</span>
              )}
              {msg.citations?.map((c, i) => (
                <span key={i} className="inline-flex items-center gap-1 rounded-lg border border-white/[0.06] bg-white/[0.03] px-2 py-0.5 font-mono text-[10px] text-dim">
                  <FiFileText className="h-2.5 w-2.5" />{c.section} p.{c.page} · {c.score.toFixed(2)}
                </span>
              ))}
            </div>
            {webSources && webSources.length > 0 && (
              <div className="flex flex-col gap-1">
                {webSources.map((s, i) => (
                  <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.06] bg-white/[0.03] px-2.5 py-1 text-[10px] text-dim transition-colors hover:border-glow/30 hover:text-glow-soft">
                    <FiExternalLink className="h-2.5 w-2.5 flex-shrink-0" />
                    <span className="font-medium">{s.source}</span>
                    <span className="truncate opacity-60">{s.title}</span>
                  </a>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
