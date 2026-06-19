/** World state query selectors. */

import type { WorldState, NPCState, EmergentQuest } from '../types.js';
import { computeViability } from '../npc/needs.js';

export function getLiveNPCs(world: WorldState): NPCState[] {
  return Object.values(world.npcs).filter(n => n.isAlive);
}

export function getTopViabilityNPCs(world: WorldState, n = 5): NPCState[] {
  return getLiveNPCs(world).sort((a, b) => computeViability(b) - computeViability(a)).slice(0, n);
}

export function getBottomViabilityNPCs(world: WorldState, n = 5): NPCState[] {
  return getLiveNPCs(world).sort((a, b) => computeViability(a) - computeViability(b)).slice(0, n);
}

export function getActiveQuests(world: WorldState): EmergentQuest[] {
  return Object.values(world.quests).filter(q => q.isActive);
}

export function getNPCsBySettlement(world: WorldState, settlementId: string): NPCState[] {
  return getLiveNPCs(world).filter(n => n.settlementId === settlementId);
}

export function getWorldSummary(world: WorldState): object {
  const liveNpcs = getLiveNPCs(world);
  const avgViability = liveNpcs.length > 0
    ? liveNpcs.reduce((sum, n) => sum + computeViability(n), 0) / liveNpcs.length
    : 0;

  return {
    day: world.day,
    settlements: Object.values(world.settlements).map(s => ({
      id: s.id, name: s.name, population: s.population,
      resources: s.resources,
    })),
    totalNPCs: liveNpcs.length,
    avgViability: avgViability.toFixed(3),
    activeQuests: getActiveQuests(world).length,
    dungeonRooms: world.dungeonRooms.length,
  };
}
