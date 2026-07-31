import type { GraphNodeT, GraphLinkT } from '@/types'

export const GRAPH_NODES: (GraphNodeT & { color: string })[] = [
  { id: 'ost', label: 'Origins Space Telescope', type: 'Paper', val: 3, color: '#6CA2C1' },
  { id: 'pope', label: 'A. Pope', type: 'Author', val: 1.5, color: '#34D399' },
  { id: 'bauer', label: 'J. Bauer', type: 'Author', val: 1.5, color: '#34D399' },
  { id: 'ir', label: 'Infrared', type: 'Keyword', val: 1.2, color: '#7DD3FC' },
  { id: 'spec', label: 'Spectroscopy', type: 'Keyword', val: 1.2, color: '#7DD3FC' },
  { id: 'exo', label: 'Exoplanets', type: 'Keyword', val: 1.2, color: '#7DD3FC' },
  { id: 'astro', label: 'Astrophysics', type: 'Domain', val: 2, color: '#E8E1D3' },
  { id: 'nasa', label: 'NASA', type: 'Institution', val: 1.8, color: '#A78BFA' },
  { id: 'sat', label: 'OST Satellite', type: 'Satellite', val: 2, color: '#60A5FA' },
  { id: 'bio', label: 'Biosignatures', type: 'Keyword', val: 1.2, color: '#7DD3FC' },
]

export const GRAPH_LINKS: GraphLinkT[] = [
  { source: 'ost', target: 'pope', label: 'AUTHORED_BY' },
  { source: 'ost', target: 'bauer', label: 'AUTHORED_BY' },
  { source: 'ost', target: 'ir', label: 'MENTIONS' },
  { source: 'ost', target: 'spec', label: 'MENTIONS' },
  { source: 'ost', target: 'exo', label: 'MENTIONS' },
  { source: 'ost', target: 'astro', label: 'IN_DOMAIN' },
  { source: 'ost', target: 'sat', label: 'DESCRIBES' },
  { source: 'sat', target: 'nasa', label: 'OPERATED_BY' },
  { source: 'exo', target: 'bio', label: 'RELATED_TO' },
  { source: 'ir', target: 'spec', label: 'RELATED_TO' },
]
