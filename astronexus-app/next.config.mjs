/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['three'],
  experimental: {
    optimizePackageImports: ['react-icons', 'framer-motion', '@react-three/drei', 'date-fns'],
  },
}
export default nextConfig
