import { NextResponse, type NextRequest } from 'next/server'
import { createServerClient } from '@supabase/ssr'

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  const next = searchParams.get('next') ?? '/dashboard'

  if (code) {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL!
    const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

    if (url && key) {
      const response = NextResponse.redirect(`${origin}${next}`)
      const supabase = createServerClient(url, key, {
        cookies: {
          getAll: () => request.cookies.getAll().map(c => ({ name: c.name, value: c.value })),
          setAll: (cookies) => cookies.forEach(({ name, value, options }) => response.cookies.set(name, value, options)),
        },
      })
      const { error } = await supabase.auth.exchangeCodeForSession(code)
      if (!error) return response
    }
  }

  return NextResponse.redirect(`${origin}/login?error=auth_callback_failed`)
}
