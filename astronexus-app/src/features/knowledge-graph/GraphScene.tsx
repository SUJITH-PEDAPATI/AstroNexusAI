/**
 * GraphScene — Interactive 3D Force-Directed Knowledge Graph
 *
 * Pure React + Canvas2D with 3D perspective projection.
 * Zero external dependencies beyond React.
 * Features:
 *   - 3D force-directed layout with depth
 *   - Mouse drag to orbit, scroll to zoom, right-drag to pan
 *   - Click nodes to highlight connected subgraph
 *   - Hover tooltips with type + connection count
 *   - Smooth 60fps animation with auto-settling
 *   - Glow effects, depth-based sizing, edge labels on hover
 */
'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

/* ── Types ─────────────────────────────────────────────────────────────── */
export interface GNode { id: string; label: string; type: string; val: number }
export interface GLink { source: string; target: string; label: string }
interface Props {
  nodes: GNode[]
  links: GLink[]
  fullScreen?: boolean
  onToggleFullScreen?: () => void
}

/* ── Colours (brighter, higher contrast for dark backgrounds) ────────── */
const TYPE_COLOR: Record<string, string> = {
  Paper:      '#60A5FA',
  Topic:      '#A78BFA',
  Author:     '#34D399',
  Keyword:    '#FBBF24',
  Entity:     '#F472B6',
  Domain:     '#2DD4BF',
  Model:      '#818CF8',
  Dataset:    '#FB923C',
  Task:       '#38BDF8',
  VoiceInput: '#E879F9',
  Query:      '#94A3B8',
  default:    '#64748B',
}
const TYPE_RADIUS: Record<string, number> = {
  Paper: 18, Topic: 16, Author: 10, Domain: 12,
  Model: 11, Dataset: 11, Entity: 9, Keyword: 8,
  Task: 9, VoiceInput: 8, Query: 8,
}

/* ── 3D math helpers ───────────────────────────────────────────────────── */
interface V3 { x: number; y: number; z: number }
function v3(x: number, y: number, z: number): V3 { return { x, y, z } }
function v3add(a: V3, b: V3): V3 { return { x: a.x+b.x, y: a.y+b.y, z: a.z+b.z } }
function v3sub(a: V3, b: V3): V3 { return { x: a.x-b.x, y: a.y-b.y, z: a.z-b.z } }
function v3scale(v: V3, s: number): V3 { return { x: v.x*s, y: v.y*s, z: v.z*s } }
function v3len(v: V3): number { return Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) }
function v3norm(v: V3): V3 { const l = v3len(v) || 1; return { x: v.x/l, y: v.y/l, z: v.z/l } }

function rotateY(p: V3, a: number): V3 {
  const c = Math.cos(a), s = Math.sin(a)
  return { x: p.x*c + p.z*s, y: p.y, z: -p.x*s + p.z*c }
}
function rotateX(p: V3, a: number): V3 {
  const c = Math.cos(a), s = Math.sin(a)
  return { x: p.x, y: p.y*c - p.z*s, z: p.y*s + p.z*c }
}

function project(p: V3, W: number, H: number, fov: number, camDist: number, panX: number, panY: number): { sx: number; sy: number; scale: number } {
  const z = p.z + camDist
  const perspective = fov / (fov + z)
  return {
    sx: W/2 + (p.x * perspective) + panX,
    sy: H/2 + (p.y * perspective) + panY,
    scale: Math.max(0.15, perspective),
  }
}

/* ── Simulation node ────────────────────────────────────────────────────── */
interface SimNode extends GNode {
  pos: V3; vel: V3
  color: string; r: number
  connCount: number
}

/* ── Force simulation (3D Fruchterman–Reingold) ─────────────────────────── */
function initSim(nodes: GNode[], links: GLink[]): SimNode[] {
  const connMap: Record<string, number> = {}
  for (const l of links) {
    connMap[l.source] = (connMap[l.source] ?? 0) + 1
    connMap[l.target] = (connMap[l.target] ?? 0) + 1
  }
  const golden = 1.618033988749895
  return nodes.map((n, i) => {
    const theta = 2 * Math.PI * i * golden
    const phi   = Math.acos(1 - 2 * (i + 0.5) / nodes.length)
    const spread = 80 + nodes.length * 2
    return {
      ...n,
      pos: v3(
        spread * Math.sin(phi) * Math.cos(theta),
        spread * Math.sin(phi) * Math.sin(theta),
        spread * Math.cos(phi),
      ),
      vel: v3(0, 0, 0),
      color: TYPE_COLOR[n.type] ?? TYPE_COLOR.default,
      r:     TYPE_RADIUS[n.type] ?? 9,
      connCount: connMap[n.id] ?? 0,
    }
  })
}

function stepSim(nodes: SimNode[], links: GLink[], cooling: number): SimNode[] {
  const N = nodes.length
  if (N === 0) return nodes
  const k = 45 + N * 0.8
  const REPEL  = k * k * 0.6
  const SPRING = 0.008
  const TARGET = k * 0.9
  const GRAVITY = 0.003
  const DAMP   = 0.88 * cooling

  const next = nodes.map(n => ({ ...n, pos: { ...n.pos }, vel: { ...n.vel } }))
  const idx  = Object.fromEntries(nodes.map((n, i) => [n.id, i]))

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = v3sub(next[i].pos, next[j].pos)
      const d2 = d.x*d.x + d.y*d.y + d.z*d.z + 1
      const f  = REPEL / d2
      const fv = v3scale(d, f / Math.sqrt(d2))
      next[i].vel = v3add(next[i].vel, fv)
      next[j].vel = v3sub(next[j].vel, fv)
    }
  }

  for (const l of links) {
    const ai = idx[l.source], bi = idx[l.target]
    if (ai === undefined || bi === undefined) continue
    const d = v3sub(next[bi].pos, next[ai].pos)
    const dist = v3len(d) + 0.01
    const f  = SPRING * (dist - TARGET)
    const fv = v3scale(v3norm(d), f)
    next[ai].vel = v3add(next[ai].vel, fv)
    next[bi].vel = v3sub(next[bi].vel, fv)
  }

  for (const n of next) {
    n.vel = v3add(n.vel, v3scale(n.pos, -GRAVITY))
    n.vel = v3scale(n.vel, DAMP)
    n.pos = v3add(n.pos, n.vel)
  }

  return next
}

/* ── Component ──────────────────────────────────────────────────────────── */
export function GraphScene({ nodes, links, fullScreen = false, onToggleFullScreen }: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [W, setW] = useState(800)
  const [H, setH] = useState(600)

  // Camera state
  const cam = useRef({ rotY: 0.3, rotX: -0.3, dist: 350, panX: 0, panY: 0 })
  const FOV = 500

  // Interaction state
  const dragRef = useRef({ active: false, button: 0, startX: 0, startY: 0, startRotY: 0, startRotX: 0, startPanX: 0, startPanY: 0 })
  const [hoverNode, setHoverNode] = useState<string | null>(null)
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; node: SimNode } | null>(null)

  // Simulation
  const simRef = useRef<SimNode[]>([])
  const stepCount = useRef(0)

  // Resize
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      setW(Math.round(e.contentRect.width) || 800)
      setH(Math.round(e.contentRect.height) || 600)
    })
    ro.observe(el)
    setW(el.clientWidth || 800)
    setH(el.clientHeight || 600)
    return () => ro.disconnect()
  }, [])

  // Init simulation when nodes change
  useEffect(() => {
    simRef.current = initSim(nodes, links)
    stepCount.current = 0
    setActiveNode(null)
    setHoverNode(null)
    setTooltip(null)
  }, [nodes, links])

  // Connected set for highlight
  const connectedSet = useCallback((nodeId: string | null): Set<string> => {
    if (!nodeId) return new Set()
    const s = new Set<string>([nodeId])
    for (const l of links) {
      if (l.source === nodeId) s.add(l.target)
      if (l.target === nodeId) s.add(l.source)
    }
    return s
  }, [links])

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf = 0
    const draw = () => {
      // Step physics (slow down after 200 steps)
      const cooling = stepCount.current < 200 ? 1.0 : Math.max(0.3, 1.0 - (stepCount.current - 200) * 0.003)
      if (stepCount.current < 600) {
        simRef.current = stepSim(simRef.current, links, cooling)
        stepCount.current++
      }

      const c = cam.current
      ctx.clearRect(0, 0, W, H)

      // Transform + project all nodes
      const projected = simRef.current.map(n => {
        let p = rotateY(n.pos, c.rotY)
        p = rotateX(p, c.rotX)
        const { sx, sy, scale } = project(p, W, H, FOV, c.dist, c.panX, c.panY)
        return { ...n, sx, sy, scale, pz: p.z }
      }).sort((a, b) => b.pz - a.pz) // depth sort

      const highlighted = connectedSet(activeNode ?? hoverNode)
      const hasHighlight = highlighted.size > 0

      // Draw edges
      const nodeScreen: Record<string, { sx: number; sy: number; scale: number }> = {}
      for (const n of projected) nodeScreen[n.id] = { sx: n.sx, sy: n.sy, scale: n.scale }

      for (const l of links) {
        const a = nodeScreen[l.source]
        const b = nodeScreen[l.target]
        if (!a || !b) continue

        const isHL = hasHighlight && highlighted.has(l.source) && highlighted.has(l.target)
        const alpha = hasHighlight ? (isHL ? 0.6 : 0.06) : 0.2

        ctx.beginPath()
        ctx.moveTo(a.sx, a.sy)
        ctx.lineTo(b.sx, b.sy)
        ctx.strokeStyle = isHL ? '#8B8BFF' : '#2A2A40'
        ctx.lineWidth   = isHL ? 1.5 : 0.7
        ctx.globalAlpha = alpha
        ctx.stroke()
        ctx.globalAlpha = 1

        // Edge label on highlight
        if (isHL && l.label) {
          const mx = (a.sx + b.sx) / 2
          const my = (a.sy + b.sy) / 2
          ctx.font      = '9px system-ui, sans-serif'
          ctx.fillStyle = '#7878A0'
          ctx.textAlign = 'center'
          ctx.fillText(l.label, mx, my - 3)
        }
      }

      // Draw nodes
      for (const n of projected) {
        const r = n.r * n.scale
        if (r < 1.5) continue

        const isActive  = activeNode === n.id
        const isHover   = hoverNode === n.id
        const isDimmed  = hasHighlight && !highlighted.has(n.id)
        const baseAlpha = isDimmed ? 0.12 : 1.0

        ctx.globalAlpha = baseAlpha

        // Glow for active/hover
        if ((isActive || isHover) && !isDimmed) {
          ctx.beginPath()
          ctx.arc(n.sx, n.sy, r + 8, 0, Math.PI * 2)
          const glow = ctx.createRadialGradient(n.sx, n.sy, r, n.sx, n.sy, r + 8)
          glow.addColorStop(0, n.color + '40')
          glow.addColorStop(1, n.color + '00')
          ctx.fillStyle = glow
          ctx.fill()
        }

        // Node circle
        ctx.beginPath()
        ctx.arc(n.sx, n.sy, r, 0, Math.PI * 2)
        const grad = ctx.createRadialGradient(n.sx - r*0.3, n.sy - r*0.3, 0, n.sx, n.sy, r)
        grad.addColorStop(0, n.color)
        grad.addColorStop(1, n.color + '99')
        ctx.fillStyle = grad
        ctx.fill()

        // Border
        if (isActive) {
          ctx.strokeStyle = '#FFFFFF'
          ctx.lineWidth = 2
          ctx.stroke()
        } else if (isHover) {
          ctx.strokeStyle = n.color
          ctx.lineWidth = 1.5
          ctx.stroke()
        }

        // Label (always visible, sized by depth — larger in fullscreen)
        if (!isDimmed && r > 4) {
          const fsBoost  = fullScreen ? 3 : 0
          const fontSize = Math.max(9 + fsBoost, Math.min(16 + fsBoost, (11 + fsBoost) * n.scale))
          ctx.font      = `500 ${fontSize}px system-ui, -apple-system, sans-serif`
          ctx.textAlign = 'center'
          ctx.fillStyle = isActive || isHover ? '#FFFFFF' : '#C8C8D8'
          ctx.globalAlpha = isDimmed ? 0.1 : (isActive || isHover ? 1 : 0.85)
          const maxLen   = fullScreen ? 30 : 22
          const label    = n.label.length > maxLen ? n.label.slice(0, maxLen - 1) + '…' : n.label
          ctx.fillText(label, n.sx, n.sy + r + fontSize + 2)
        }

        ctx.globalAlpha = 1
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [W, H, links, activeNode, hoverNode, connectedSet, nodes])

  // Hit test helper
  const hitTest = useCallback((mx: number, my: number): SimNode | null => {
    const c = cam.current
    let closest: SimNode | null = null
    let closestDist = Infinity
    for (const n of simRef.current) {
      let p = rotateY(n.pos, c.rotY)
      p = rotateX(p, c.rotX)
      const { sx, sy, scale } = project(p, W, H, FOV, c.dist, c.panX, c.panY)
      const r = n.r * scale
      const dx = mx - sx, dy = my - sy
      const d2 = dx*dx + dy*dy
      if (d2 < (r + 6) * (r + 6) && d2 < closestDist) {
        closestDist = d2; closest = n
      }
    }
    return closest
  }, [W, H])

  // Mouse handlers
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    dragRef.current = {
      active: true, button: e.button,
      startX: e.clientX, startY: e.clientY,
      startRotY: cam.current.rotY, startRotX: cam.current.rotX,
      startPanX: cam.current.panX, startPanY: cam.current.panY,
    }
  }, [])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top

    if (dragRef.current.active) {
      const dx = e.clientX - dragRef.current.startX
      const dy = e.clientY - dragRef.current.startY
      if (dragRef.current.button === 2 || e.shiftKey) {
        cam.current.panX = dragRef.current.startPanX + dx
        cam.current.panY = dragRef.current.startPanY + dy
      } else {
        cam.current.rotY = dragRef.current.startRotY + dx * 0.005
        cam.current.rotX = dragRef.current.startRotX + dy * 0.005
        cam.current.rotX = Math.max(-1.2, Math.min(1.2, cam.current.rotX))
      }
    } else {
      const hit = hitTest(mx, my)
      setHoverNode(hit?.id ?? null)
      if (hit) {
        let p = rotateY(hit.pos, cam.current.rotY)
        p = rotateX(p, cam.current.rotX)
        const pr = project(p, W, H, FOV, cam.current.dist, cam.current.panX, cam.current.panY)
        setTooltip({ x: pr.sx + rect.left, y: pr.sy + rect.top - 20, node: hit })
      } else {
        setTooltip(null)
      }
    }
  }, [hitTest, W, H])

  const onMouseUp = useCallback((e: React.MouseEvent) => {
    const dx = Math.abs(e.clientX - dragRef.current.startX)
    const dy = Math.abs(e.clientY - dragRef.current.startY)
    if (dx < 4 && dy < 4) {
      const rect = canvasRef.current!.getBoundingClientRect()
      const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top)
      setActiveNode(prev => prev === hit?.id ? null : hit?.id ?? null)
    }
    dragRef.current.active = false
  }, [hitTest])

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    cam.current.dist = Math.max(100, Math.min(1200, cam.current.dist + e.deltaY * 0.5))
  }, [])

  // Zoom control methods exposed via ref
  const zoomIn  = useCallback(() => { cam.current.dist = Math.max(100, cam.current.dist - 40) }, [])
  const zoomOut = useCallback(() => { cam.current.dist = Math.min(1200, cam.current.dist + 40) }, [])
  const resetView = useCallback(() => {
    cam.current = { rotY: 0.3, rotX: -0.3, dist: 350, panX: 0, panY: 0 }
    stepCount.current = 0
    simRef.current = initSim(nodes, links)
  }, [nodes, links])

  if (nodes.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <p className="text-sm text-dim">No graph data yet.</p>
        <p className="text-xs text-faint">Ingest a paper or ask a query to build the graph.</p>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        width={W} height={H}
        className="h-full w-full cursor-grab active:cursor-grabbing"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { dragRef.current.active = false; setHoverNode(null); setTooltip(null) }}
        onWheel={onWheel}
        onContextMenu={e => e.preventDefault()}
      />

      {/* Controls cluster */}
      <div className="absolute right-4 bottom-16 flex flex-col gap-1.5">
        <button onClick={zoomIn} className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-[#0A0A12]/80 text-sm text-silver-light backdrop-blur-xl hover:bg-white/[0.08] active:bg-white/[0.12]" aria-label="Zoom in">+</button>
        <button onClick={zoomOut} className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-[#0A0A12]/80 text-sm text-silver-light backdrop-blur-xl hover:bg-white/[0.08] active:bg-white/[0.12]" aria-label="Zoom out">−</button>
        <button onClick={resetView} className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-[#0A0A12]/80 text-[9px] text-silver-light backdrop-blur-xl hover:bg-white/[0.08] active:bg-white/[0.12]" aria-label="Reset view">⟲</button>
        {onToggleFullScreen && (
          <button onClick={onToggleFullScreen} className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-[#0A0A12]/80 text-sm text-silver-light backdrop-blur-xl hover:bg-white/[0.08] active:bg-white/[0.12]" aria-label={fullScreen ? 'Exit full screen' : 'Full screen'}>
            {fullScreen ? '⛶' : '⛶'}
          </button>
        )}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 rounded-xl border border-white/[0.12] bg-[#0C0C14]/95 px-3.5 py-2.5 shadow-2xl backdrop-blur-xl"
          style={{ left: tooltip.x, top: tooltip.y, transform: 'translate(-50%, -100%) translateY(-8px)' }}
        >
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: tooltip.node.color }} />
            <span className="text-[13px] font-semibold text-white">{tooltip.node.label}</span>
          </div>
          <div className="mt-1 flex gap-3 text-[10px]">
            <span className="text-faint">Type: <span className="text-silver-light">{tooltip.node.type}</span></span>
            <span className="text-faint">Connections: <span className="text-silver-light">{tooltip.node.connCount}</span></span>
          </div>
        </div>
      )}
    </div>
  )
}
