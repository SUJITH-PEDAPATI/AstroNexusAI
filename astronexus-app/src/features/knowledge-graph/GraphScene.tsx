'use client'
import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { GRAPH_NODES, GRAPH_LINKS } from './data'

interface Positioned { id: string; label: string; color: string; val: number; pos: THREE.Vector3 }

function useLayout(): Positioned[] {
  return useMemo(() => {
    // Deterministic spherical layout
    const n = GRAPH_NODES.length
    return GRAPH_NODES.map((node, i) => {
      const phi = Math.acos(-1 + (2 * i) / n)
      const theta = Math.sqrt(n * Math.PI) * phi
      const r = 3.2
      return {
        id: node.id, label: node.label, color: node.color, val: node.val ?? 1,
        pos: new THREE.Vector3(r * Math.cos(theta) * Math.sin(phi), r * Math.sin(theta) * Math.sin(phi), r * Math.cos(phi)),
      }
    })
  }, [])
}

function Node({ node, active, onClick }: { node: Positioned; active: boolean; onClick: () => void }) {
  const ref = useRef<THREE.Mesh>(null)
  const [hover, setHover] = useState(false)
  useFrame((state) => {
    if (ref.current) {
      const s = 1 + Math.sin(state.clock.elapsedTime * 2 + node.pos.x) * 0.04
      ref.current.scale.setScalar((hover || active ? 1.35 : 1) * s)
    }
  })
  return (
    <group position={node.pos}>
      <mesh ref={ref} onClick={onClick} onPointerOver={() => setHover(true)} onPointerOut={() => setHover(false)}>
        <sphereGeometry args={[0.12 * node.val, 24, 24]} />
        <meshStandardMaterial color={node.color} emissive={node.color} emissiveIntensity={hover || active ? 1.4 : 0.6} toneMapped={false} />
      </mesh>
      {(hover || active) && (
        <Html center distanceFactor={10}>
          <div className="pointer-events-none whitespace-nowrap rounded-lg border border-white/10 bg-void/90 px-2 py-1 text-[10px] text-light backdrop-blur">{node.label}</div>
        </Html>
      )}
    </group>
  )
}

function Graph() {
  const nodes = useLayout()
  const [active, setActive] = useState<string | null>(null)
  const byId = useMemo(() => Object.fromEntries(nodes.map((n) => [n.id, n])), [nodes])
  const group = useRef<THREE.Group>(null)
  useFrame((_, d) => { if (group.current) group.current.rotation.y += d * 0.04 })

  return (
    <group ref={group}>
      {GRAPH_LINKS.map((l, i) => {
        const a = byId[l.source]; const b = byId[l.target]
        if (!a || !b) return null
        const lit = active === l.source || active === l.target
        return <Line key={i} points={[a.pos, b.pos]} color={lit ? '#7DD3FC' : '#3A4048'} lineWidth={lit ? 2 : 1} transparent opacity={lit ? 0.9 : 0.4} />
      })}
      {nodes.map((n) => <Node key={n.id} node={n} active={active === n.id} onClick={() => setActive(active === n.id ? null : n.id)} />)}
    </group>
  )
}

export function GraphScene() {
  return (
    <Canvas camera={{ position: [0, 0, 9], fov: 55 }} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1.2} />
      <Graph />
      <OrbitControls enablePan enableZoom enableRotate minDistance={4} maxDistance={16} autoRotate={false} />
    </Canvas>
  )
}
