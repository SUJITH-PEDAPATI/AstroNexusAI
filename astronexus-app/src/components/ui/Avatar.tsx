import { cn } from '@/lib/cn'
export function Avatar({ name, color, size = 36 }: { name: string; color: string; size?: number }) {
  const initials = name.split(' ').map((n) => n[0]).slice(0, 2).join('').toUpperCase()
  return (
    <span className={cn('inline-flex items-center justify-center rounded-full font-semibold text-white')}
      style={{ width: size, height: size, background: `linear-gradient(135deg, ${color}, ${color}99)`, fontSize: size * 0.36 }}>
      {initials}
    </span>
  )
}
