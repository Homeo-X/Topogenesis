/** NPC relationship management. */

import type { RelationshipState, KinshipType } from '../types.js';

export function createRelationship(targetId: string, kinshipType: KinshipType = 'stranger'): RelationshipState {
  return {
    targetId,
    familiarity: kinshipType === 'kin' || kinshipType === 'spouse' ? 0.7 : 0.0,
    affection: kinshipType === 'kin' ? 0.4 : kinshipType === 'enemy' ? -0.5 : 0.0,
    trust: kinshipType === 'kin' ? 0.6 : kinshipType === 'enemy' ? -0.4 : 0.1,
    respect: 0.2,
    conflict: kinshipType === 'rival' || kinshipType === 'enemy' ? 0.5 : 0.0,
    kinshipType,
    romanceInterest: 0,
    obligationStrength: kinshipType === 'kin' ? 0.5 : 0,
    lastInteractionDay: 0,
  };
}

export type InteractionType = 'gift' | 'help' | 'work_together' | 'social' | 'argument' | 'betrayal' | 'apology' | 'festival' | 'healing' | 'trade';

/** Update a relationship after an interaction. Returns updated relationship. */
export function applyInteraction(
  rel: RelationshipState,
  type: InteractionType,
  day: number,
  quality: number = 0.5, // 0–1
): RelationshipState {
  const DELTAS: Record<InteractionType, Partial<RelationshipState>> = {
    gift:          { affection: 0.15, trust: 0.05, familiarity: 0.05 },
    help:          { affection: 0.10, trust: 0.10, familiarity: 0.10, respect: 0.05 },
    work_together: { familiarity: 0.08, respect: 0.08, trust: 0.05 },
    social:        { familiarity: 0.06, affection: 0.05, belonging: 0.04 } as any,
    argument:      { conflict: 0.20, affection: -0.10, trust: -0.05 },
    betrayal:      { conflict: 0.40, trust: -0.30, affection: -0.20 },
    apology:       { conflict: -0.15, affection: 0.05, trust: 0.03 },
    festival:      { affection: 0.08, familiarity: 0.06, conflict: -0.05 },
    healing:       { affection: 0.12, trust: 0.12, obligationStrength: 0.10 },
    trade:         { familiarity: 0.05, respect: 0.05, trust: 0.03 },
  };

  const deltas = DELTAS[type];
  const scale = quality;

  return {
    ...rel,
    familiarity: clamp((rel.familiarity) + ((deltas.familiarity ?? 0) * scale), 0, 1),
    affection: clamp(rel.affection + ((deltas.affection ?? 0) * scale), -1, 1),
    trust: clamp(rel.trust + ((deltas.trust ?? 0) * scale), -1, 1),
    respect: clamp(rel.respect + ((deltas.respect ?? 0) * scale), 0, 1),
    conflict: clamp(rel.conflict + ((deltas.conflict ?? 0) * scale), 0, 1),
    obligationStrength: clamp(rel.obligationStrength + ((deltas.obligationStrength ?? 0) * scale), 0, 1),
    lastInteractionDay: day,
  };
}

/** Drift relationships that haven't been interacted with recently. */
export function driftRelationship(rel: RelationshipState, currentDay: number): RelationshipState {
  const daysSince = currentDay - rel.lastInteractionDay;
  if (daysSince < 5) return rel;
  const driftRate = 0.005 * Math.min(daysSince, 30);
  return {
    ...rel,
    familiarity: Math.max(0, rel.familiarity - driftRate),
    affection: rel.affection > 0 ? Math.max(0, rel.affection - driftRate) : Math.min(0, rel.affection + driftRate * 0.5),
    conflict: Math.max(0, rel.conflict - driftRate),
  };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/** Get the most socially valuable relationship target from a set. */
export function strongestRelationship(relationships: Record<string, RelationshipState>): RelationshipState | undefined {
  const rels = Object.values(relationships);
  if (rels.length === 0) return undefined;
  return rels.reduce((best, r) => {
    const score = r.familiarity + r.affection + r.trust;
    const bestScore = best.familiarity + best.affection + best.trust;
    return score > bestScore ? r : best;
  });
}
