import type { Config } from 'tailwindcss'

/**
 * Design tokens derived directly from the uploaded cinematic hero video:
 * deep space black, NASA blue, steel-cyan atmospheric glow, metallic gray,
 * and soft warm-white — a moody mission-control identity.
 */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#07080A',
        space: '#0A0E14',
        surface: '#10151E',
        'surface-2': '#161C28',
        panel: '#1B2331',
        nasa: { DEFAULT: '#0B3D91', light: '#1D4ED8' },
        blue: { DEFAULT: '#3B82F6', bright: '#60A5FA', deep: '#2563EB' },
        glow: { DEFAULT: '#6CA2C1', soft: '#7DD3FC', bright: '#A5D8F0' },
        steel: { DEFAULT: '#6C6F72', light: '#9CA3AF', dark: '#3A4048' },
        gold: { DEFAULT: '#E8E1D3', deep: '#D4B483' },
        light: '#F5F7FA',
        dim: '#94A3B8',
        faint: '#5B6675',
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        body: ['var(--font-body)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      animation: {
        'glow-pulse': 'glowPulse 4s ease-in-out infinite',
        'float-slow': 'floatSlow 8s ease-in-out infinite',
        shimmer: 'shimmer 2.5s linear infinite',
        'spin-slow': 'spin 24s linear infinite',
      },
      keyframes: {
        glowPulse: { '0%,100%': { opacity: '0.35' }, '50%': { opacity: '0.75' } },
        floatSlow: { '0%,100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } },
        shimmer: { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
      },
    },
  },
  plugins: [],
} satisfies Config
