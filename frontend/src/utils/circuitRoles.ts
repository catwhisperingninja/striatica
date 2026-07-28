// frontend/src/utils/circuitRoles.ts
// Role handling that generalizes the legacy source/intermediate/sink taxonomy
// to arbitrary traced-circuit role strings, while keeping legacy circuits
// (gpt2 co-activation/similarity) rendering pixel-identical to before.
import type { CircuitNode } from '../types/circuit'
import { COLORS, TRACED_CIRCUITS } from '../config/rendering'

/** The three legacy roles, in their canonical display order. */
export const LEGACY_ROLES = ['source', 'intermediate', 'sink'] as const

const LEGACY_ROLE_SET: ReadonlySet<string> = new Set(LEGACY_ROLES)

/** Human-readable label for a role. Legacy roles map to their historical names;
 *  any other role string is shown verbatim. */
export function roleLabel(role: string): string {
  switch (role) {
    case 'source':
      return 'Source'
    case 'intermediate':
      return 'Processing'
    case 'sink':
      return 'Output'
    default:
      return role
  }
}

/**
 * Map each role present in `nodes` to a color hex string.
 *
 * - Legacy roles → the exact existing COLORS.roles / COLORS.roles3D values, so
 *   gpt2 circuits render pixel-identical.
 * - 'unassigned' → the neutral TRACED_CIRCUITS unassigned color.
 * - Any other role → assigned from TRACED_CIRCUITS.rolePalette(3D) by its index
 *   in the lexicographically-sorted list of non-legacy roles (deterministic,
 *   stable across renders regardless of node order).
 */
export function buildRoleColorMap(nodes: CircuitNode[], space: '2d' | '3d'): Map<string, string> {
  const legacy = space === '2d' ? COLORS.roles : COLORS.roles3D
  const palette = space === '2d' ? TRACED_CIRCUITS.rolePalette : TRACED_CIRCUITS.rolePalette3D
  const unassigned = space === '2d' ? TRACED_CIRCUITS.unassignedColor : TRACED_CIRCUITS.unassignedColor3D

  const roles = new Set<string>()
  for (const n of nodes) roles.add(n.role)

  const map = new Map<string, string>()
  const other: string[] = []
  for (const role of roles) {
    if (LEGACY_ROLE_SET.has(role)) {
      map.set(role, legacy[role])
    } else if (role === 'unassigned') {
      map.set(role, unassigned)
    } else {
      other.push(role)
    }
  }

  other.sort()
  other.forEach((role, i) => {
    map.set(role, palette[i % palette.length])
  })

  return map
}

/**
 * The order in which roles should be listed. If every role is legacy, use the
 * fixed legacy order (source → intermediate → sink). Otherwise order by node
 * count descending, with 'unassigned' always last.
 */
export function orderedRoles(nodes: CircuitNode[]): string[] {
  const counts = new Map<string, number>()
  for (const n of nodes) counts.set(n.role, (counts.get(n.role) ?? 0) + 1)

  const roles = [...counts.keys()]
  const allLegacy = roles.every((r) => LEGACY_ROLE_SET.has(r))
  if (allLegacy) {
    return LEGACY_ROLES.filter((r) => counts.has(r))
  }

  return roles.sort((a, b) => {
    if (a === 'unassigned') return 1
    if (b === 'unassigned') return -1
    const d = (counts.get(b) ?? 0) - (counts.get(a) ?? 0)
    return d !== 0 ? d : a.localeCompare(b)
  })
}
