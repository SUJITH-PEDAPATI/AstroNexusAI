'use client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import { memo } from 'react'

/** Renders ChatGPT-quality markdown: GFM tables, LaTeX math, highlighted code. */
export const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-anx">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})
