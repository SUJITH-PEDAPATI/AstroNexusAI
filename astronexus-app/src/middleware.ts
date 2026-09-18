import { NextResponse } from 'next/server'

// Authentication removed — all routes are publicly accessible.
export function middleware() {
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico|videos|images|textures|models|icons).*)'],
}
