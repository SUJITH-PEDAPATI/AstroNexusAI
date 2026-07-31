import { NextResponse, type NextRequest } from 'next/server'

/**
 * Route protection. The (app) group requires an authenticated session. We check
 * a lightweight session cookie so this works at the edge with zero backend.
 * Swap the cookie check for NextAuth's `auth()` middleware when wiring real auth.
 */
const PROTECTED = [
  '/dashboard', '/chat', '/research', '/knowledge-graph',
  '/vision-ai', '/papers', '/profile', '/settings',
]
const AUTH_ROUTES = ['/login', '/register']
const SESSION_COOKIE = 'anx_session'

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl
  const hasSession = req.cookies.has(SESSION_COOKIE)

  if (PROTECTED.some((p) => pathname === p || pathname.startsWith(p + '/'))) {
    if (!hasSession) {
      const url = new URL('/login', req.url)
      url.searchParams.set('from', pathname)
      return NextResponse.redirect(url)
    }
  }
  if (AUTH_ROUTES.includes(pathname) && hasSession) {
    return NextResponse.redirect(new URL('/dashboard', req.url))
  }
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico|videos|images|textures|models|icons).*)'],
}
