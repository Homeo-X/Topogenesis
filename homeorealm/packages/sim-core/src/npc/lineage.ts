/** Household lineage tracking. */

import type { LineageRecord } from '../types.js';

export function addLineageRecord(
  lineage: LineageRecord[],
  npcId: string,
  name: string,
  birthDay: number,
  role: string,
): LineageRecord[] {
  return [...lineage, { npcId, name, birthDay, role }];
}

export function markLineageDeath(lineage: LineageRecord[], npcId: string, deathDay: number): LineageRecord[] {
  return lineage.map(r => r.npcId === npcId ? { ...r, deathDay } : r);
}

export function getLivingLineageMembers(lineage: LineageRecord[]): LineageRecord[] {
  return lineage.filter(r => r.deathDay === undefined);
}
