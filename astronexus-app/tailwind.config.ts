import type { Config } from 'tailwindcss'

/**
 * AstroNexus AI — Monochrome Design System
 *
 * Premium graphite/charcoal palette inspired by OpenAI, Linear, Raycast,
 * Vercel, and Apple Pro apps. Every value reads from a CSS variable defined
 * in globals.css, so the whole theme can be retuned in one place.
 *
 * Token NAMES are unchanged from the previous palette — every existing
 * component re-skins automatically with zero code changes.
 */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // ── Surfaces (deep black → graphite) ──
        void:        'var(--bg-primary)',      // #050505
        space:       'var(--bg-secondary)',    // #0C0C0C
        surface:     'var(--surface)',         // #141414
        'surface-2': 'var(--surface-2)',       // #1A1A1A
        panel:       'var(--panel)',           // #212121

        // ── Interactive (graphite, was NASA blue) ──
        nasa: {
          DEFAULT: 'var(--graphite)',          // #2C2C2C
          light:   'var(--graphite-light)',    // #3A3A3A
        },
        blue: {
          DEFAULT: 'var(--button-bg)',         // #2E2E2E — primary buttons
          bright:  'var(--button-hover)',      // #3D3D3D
          deep:    'var(--button-active)',     // #1D1D1D
        },

        // ── Accents (silver, was steel-cyan) ──
        glow: {
          DEFAULT: 'var(--silver)',            // #8A8A8A
          soft:    'var(--silver-light)',      // #B4B4B4
          bright:  'var(--silver-bright)',     // #D8D8D8
        },
        steel: {
          DEFAULT: 'var(--steel)',             // #6E6E6E
          light:   'var(--steel-light)',       // #9A9A9A
          dark:    'var(--steel-dark)',        // #3A3A3A
        },
        gold: {
          DEFAULT: 'var(--warm-white)',        // #E8E6E3
          deep:    'var(--warm-gray)',         // #A8A49E
        },

        // ── Silver text utilities ──
        silver: {
          DEFAULT: 'var(--silver)',
          light:   'var(--silver-light)',
          bright:  'var(--silver-bright)',
        },

        // ── Text ──
        light: 'var(--text-primary)',          // #F2F2F2
        dim:   'var(--text-secondary)',        // #9A9A9A
        faint: 'var(--text-tertiary)',         // #626262
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        body:    ['var(--font-body)',    'system-ui', 'sans-serif'],
        mono:    ['var(--font-mono)',    'monospace'],
      },
      borderRadius: {
        card:   '16px',
        button: '16px',
      },
      animation: {
        'glow-pulse': 'glowPulse 6s ease-in-out infinite',
        'float-slow': 'floatSlow 8s ease-in-out infinite',
        shimmer:      'shimmer 2.5s linear infinite',
        'spin-slow':  'spin 24s linear infinite',
        'bg-drift':   'bgDrift 52s ease-in-out infinite',
        'bg-breathe': 'bgBreathe 44s ease-in-out infinite',
      },
      keyframes: {
        glowPulse: { '0%,100%': { opacity: '0.25' }, '50%': { opacity: '0.5' } },
        floatSlow: { '0%,100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } },
        shimmer:   { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
        // Extremely slow background motion — position + size only, never rotation
        bgDrift: {
          '0%,100%': { backgroundPosition: '0% 0%,   100% 100%, 50% 50%' },
          '50%':     { backgroundPosition: '100% 50%, 0% 0%,     50% 60%' },
        },
        bgBreathe: {
          '0%,100%': { backgroundSize: '160% 160%, 140% 140%, 200% 200%', opacity: '1' },
          '50%':     { backgroundSize: '180% 180%, 160% 160%, 210% 210%', opacity: '0.92' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config
