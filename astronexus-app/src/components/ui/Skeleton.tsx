import { cn } from '@/lib/cn'
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-lg bg-white/[0.05]', className)} />
}
