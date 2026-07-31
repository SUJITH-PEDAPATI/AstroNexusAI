import type { Variants } from 'framer-motion'

export const EASE = {
  out: [0.16, 1, 0.3, 1] as [number, number, number, number],
  inOut: [0.65, 0, 0.35, 1] as [number, number, number, number],
}

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 28 },
  visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.07, duration: 0.7, ease: EASE.out } }),
}
export const blurReveal: Variants = {
  hidden: { opacity: 0, filter: 'blur(10px)', y: 16 },
  visible: (i = 0) => ({ opacity: 1, filter: 'blur(0px)', y: 0, transition: { delay: i * 0.07, duration: 0.8, ease: EASE.out } }),
}
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: (i = 0) => ({ opacity: 1, scale: 1, transition: { delay: i * 0.06, duration: 0.6, ease: EASE.out } }),
}
export const stagger: Variants = { hidden: {}, visible: { transition: { staggerChildren: 0.07 } } }
