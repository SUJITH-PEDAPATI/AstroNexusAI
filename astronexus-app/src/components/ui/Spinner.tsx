export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <span className="inline-block animate-spin rounded-full border-2 border-white/15 border-t-glow"
      style={{ width: size, height: size }} aria-label="Loading" />
  )
}
