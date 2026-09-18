import { NextResponse, type NextRequest } from 'next/server'
import { createServerClient } from '@supabase/ssr'

/**
 * OAuth callback handler.
 *
 * Flow: Provider → redirect here with ?code= → exchange for session → set cookies → redirect.
 *
 * Critical: this MUST set BOTH the Supabase auth cookies AND the anx_session cookie.
 * The middleware checks anx_session to decide whether to allow protected routes.
 * Without it, the user gets redirected back to /login immediately after OAuth succeeds.
 */
export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url)
  const code  = searchParams.get('code')
  const next  = searchParams.get('next') ?? '/dashboard'
  const error = searchParams.get('error')
  const errorDesc = searchParams.get('error_description')

  // Handle OAuth errors (user cancelled, provider error, etc.)
  if (error) {
    const msg = errorDesc || error
    return NextResponse.redirect(
      `${origin}/login?error=${encodeURIComponent(msg)}`,
    )
  }

  if (!code) {
    return NextResponse.redirect(`${origin}/login?error=${encodeURIComponent('No authorization code received')}`)
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!url || !key) {
    return NextResponse.redirect(`${origin}/login?error=${encodeURIComponent('Supabase not configured')}`)
  }

  const response = NextResponse.redirect(`${origin}${next}`)

  const supabase = createServerClient(url, key, {
    cookies: {
      getAll: () =>
        request.cookies.getAll().map(c => ({ name: c.name, value: c.value })),
      setAll: (cookies) =>
        cookies.forEach(({ name, value, options }) =>
          response.cookies.set(name, value, options),
        ),
    },
  })

  const { data, error: exchangeError } = await supabase.auth.exchangeCodeForSession(code)

  if (exchangeError) {
    console.error('[Auth Callback] Exchange failed:', exchangeError.message)
    return NextResponse.redirect(
      `${origin}/login?error=${encodeURIComponent(exchangeError.message)}`,
    )
  }

  // ── CRITICAL: Set the anx_session cookie that middleware checks ──────
  // Without this, the middleware blocks /dashboard and redirects to /login.
  if (data.session?.user?.id) {
    response.cookies.set('anx_session', data.session.user.id, {
      path:     '/',
      httpOnly: false,        // client JS needs to read it for AuthGate
      secure:   process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge:   60 * 60 * 24 * 30, // 30 days
    })
  }

  return response
}
