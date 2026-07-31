'use client'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useRef, useEffect } from 'react'
import * as THREE from 'three'
import { StarField } from './StarField'
import { useDeviceTier } from '@/hooks/useDeviceTier'
import { useReducedMotion } from '@/hooks/useReducedMotion'

function Drift() {
  const { camera } = useThree()
  const target = useRef(new THREE.Vector3(0, 0, 6))
  const mouse = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1
      mouse.current.y = (e.clientY / window.innerHeight) * 2 - 1
    }
    window.addEventListener('mousemove', onMove, { passive: true })
    return () => window.removeEventListener('mousemove', onMove)
  }, [])

  useFrame((state, d) => {
    const t = state.clock.elapsedTime
    target.current.set(
      mouse.current.x * 0.5 + Math.sin(t * 0.08) * 0.12,
      -mouse.current.y * 0.3 + Math.cos(t * 0.1) * 0.08,
      6,
    )
    camera.position.lerp(target.current, 1 - Math.pow(0.02, d))
    camera.lookAt(0, 0, 0)
  })
  return null
}

/** Ambient starfield behind the whole app. Lazy-loaded, client-only. */
export function SpaceCanvas() {
  const tier = useDeviceTier()
  const reduced = useReducedMotion()
  return (
    <Canvas className="!fixed inset-0" dpr={tier === 'high' ? [1, 2] : [1, 1.5]}
      camera={{ position: [0, 0, 6], fov: 60 }}
      gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}>
      <StarField tier={tier} />
      {!reduced && <Drift />}
    </Canvas>
  )
}
