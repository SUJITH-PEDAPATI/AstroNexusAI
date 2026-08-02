'use client'

/**
 * AtmosphereGlow — monochrome layered background.
 *
 * Renders the animated graphite gradient composition defined in globals.css
 * (`.anx-bg`). Three composited gradient layers + noise + vignette, animating
 * only background-position and background-size over 44–52s.
 *
 * Pure CSS. No canvas, no WebGL, no video. GPU-composited, 60 FPS.
 */
export function AtmosphereGlow() {
  return <div className="anx-bg" aria-hidden />
}
