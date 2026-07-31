'use client'
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { DeviceTier } from '@/types'

const COUNTS: Record<DeviceTier, number> = { low: 800, mid: 2000, high: 3400 }

export function StarField({ tier }: { tier: DeviceTier }) {
  const near = useRef<THREE.Points>(null)
  const far = useRef<THREE.Points>(null)
  const n = COUNTS[tier]

  const make = (count: number, spread: number) => {
    const pos = new Float32Array(count * 3)
    const col = new Float32Array(count * 3)
    const c = new THREE.Color()
    for (let i = 0; i < count; i++) {
      const r = spread * (0.55 + Math.random() * 0.45)
      const th = Math.random() * Math.PI * 2
      const ph = Math.acos(2 * Math.random() - 1)
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th)
      pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th)
      pos[i * 3 + 2] = r * Math.cos(ph)
      c.setHSL(0.55 + Math.random() * 0.08, 0.3, 0.6 + Math.random() * 0.4)
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b
    }
    return { pos, col }
  }

  const a = useMemo(() => make(Math.floor(n * 0.35), 42), [n])
  const b = useMemo(() => make(Math.floor(n * 0.65), 82), [n])

  useFrame((_, d) => {
    if (near.current) near.current.rotation.y += d * 0.007
    if (far.current) far.current.rotation.y += d * 0.0025
  })

  return (
    <>
      <points ref={near}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[a.pos, 3]} />
          <bufferAttribute attach="attributes-color" args={[a.col, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.2} sizeAttenuation vertexColors transparent opacity={0.9}
          depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
      <points ref={far}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[b.pos, 3]} />
          <bufferAttribute attach="attributes-color" args={[b.col, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.09} sizeAttenuation vertexColors transparent opacity={0.5}
          depthWrite={false} blending={THREE.AdditiveBlending} />
      </points>
    </>
  )
}
