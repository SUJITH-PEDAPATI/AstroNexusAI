import type { GraphNodeT, GraphLinkT } from '@/types'

export const GRAPH_NODES: (GraphNodeT & { color: string })[] = [
  { id: 'ost', label: 'Origins Space Telescope', type: 'Paper', val: 3, color: '#8A8A8A' },
  { id: 'pope', label: 'A. Pope', type: 'Author', val: 1.5, color: '#9A9A9A' },
  { id: 'bauer', label: 'J. Bauer', type: 'Author', val: 1.5, color: '#9A9A9A' },
  { id: 'ir', label: 'Infrared', type: 'Keyword', val: 1.2, color: '#B4B4B4' },
  { id: 'spec', label: 'Spectroscopy', type: 'Keyword', val: 1.2, color: '#B4B4B4' },
  { id: 'exo', label: 'Exoplanets', type: 'Keyword', val: 1.2, color: '#B4B4B4' },
  { id: 'astro', label: 'Astrophysics', type: 'Domain', val: 2, color: '#D8D8D8' },
  { id: 'nasa', label: 'NASA', type: 'Institution', val: 1.8, color: '#7E7E7E' },
  { id: 'sat', label: 'OST Satellite', type: 'Satellite', val: 2, color: '#9A9A9A' },
  { id: 'bio', label: 'Biosignatures', type: 'Keyword', val: 1.2, color: '#B4B4B4' },
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
