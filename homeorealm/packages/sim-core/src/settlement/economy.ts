/** Settlement resource economy: production, consumption, and conservation. */

import type { SettlementResources, SettlementState, NPCState } from '../types.js';
import type { SeasonalModifiers } from '../time.js';

const DAILY_CONSUMPTION_PER_NPC: Partial<SettlementResources> = {
  food: 0.18,
  medicine: 0.02,
  tools: 0.01,
};

const RESOURCE_CAPS: SettlementResources = {
  food: 200, wood: 150, ore: 100, cloth: 80,
  medicine: 60, tools: 80, coin: 1000,
  heartwellStability: 1, publicMorale: 1, security: 1,
};

const RESOURCE_REGEN: Partial<SettlementResources> = {
  // base natural regeneration per day (small amounts)
  wood: 0.5, ore: 0.1,
};

/** Apply daily settlement-level consumption based on population. */
export function applyDailyConsumption(
  resources: SettlementResources,
  populationCount: number,
  seasonal: SeasonalModifiers,
): { resources: SettlementResources; shortages: string[] } {
  const shortages: string[] = [];
  const updated = { ...resources };

  const foodNeeded = (DAILY_CONSUMPTION_PER_NPC.food ?? 0) * populationCount;
  if (updated.food >= foodNeeded) {
    updated.food = Math.max(0, updated.food - foodNeeded);
  } else {
    shortages.push('food');
    updated.food = 0;
    updated.publicMorale = Math.max(0, updated.publicMorale - 0.05);
    updated.heartwellStability = Math.max(0, updated.heartwellStability - 0.02);
  }

  const medNeeded = (DAILY_CONSUMPTION_PER_NPC.medicine ?? 0) * populationCount * seasonal.illnessRisk;
  updated.medicine = Math.max(0, updated.medicine - medNeeded);
  if (updated.medicine < 2) shortages.push('medicine');

  const toolsNeeded = (DAILY_CONSUMPTION_PER_NPC.tools ?? 0) * populationCount;
  updated.tools = Math.max(0, updated.tools - toolsNeeded);
  if (updated.tools < 2) shortages.push('tools');

  // Morale and security drift slowly toward 0.5 without active maintenance
  updated.publicMorale = drift(updated.publicMorale, 0.5, 0.01);
  updated.security = drift(updated.security, 0.5, 0.005);
  updated.heartwellStability = drift(updated.heartwellStability, 0.7, 0.005);

  // Natural resource regen
  for (const [key, rate] of Object.entries(RESOURCE_REGEN)) {
    const k = key as keyof SettlementResources;
    const cap = RESOURCE_CAPS[k] as number;
    updated[k] = Math.min(cap, (updated[k] as number) + (rate ?? 0) * seasonal.resourceRegenRate) as any;
  }

  return { resources: clampResources(updated), shortages };
}

/** Apply resource deltas from NPC actions to settlement. */
export function applyResourceDelta(
  resources: SettlementResources,
  deltas: Partial<SettlementResources>,
): SettlementResources {
  const updated = { ...resources };
  for (const [key, delta] of Object.entries(deltas)) {
    if (delta === undefined) continue;
    const k = key as keyof SettlementResources;
    const cap = RESOURCE_CAPS[k] as number;
    (updated as any)[k] = Math.min(cap, Math.max(0, (updated[k] as number) + delta));
  }
  return updated;
}

/** Get a human-readable pressure summary for a settlement. */
export function getSettlementPressures(
  resources: SettlementResources,
  population: number,
): string[] {
  const pressures: string[] = [];
  const daysOfFood = population > 0 ? resources.food / (population * 0.3) : 999;
  if (daysOfFood < 5) pressures.push('critical_food_shortage');
  else if (daysOfFood < 15) pressures.push('food_shortage');
  if (resources.medicine < 5) pressures.push('medicine_shortage');
  if (resources.tools < 5) pressures.push('tools_shortage');
  if (resources.security < 0.3) pressures.push('security_crisis');
  if (resources.publicMorale < 0.3) pressures.push('low_morale');
  if (resources.heartwellStability < 0.4) pressures.push('hearthwell_instability');
  return pressures;
}

function drift(value: number, target: number, rate: number): number {
  return value + (target - value) * rate;
}

function clampResources(r: SettlementResources): SettlementResources {
  const result = { ...r };
  for (const key of Object.keys(RESOURCE_CAPS) as Array<keyof SettlementResources>) {
    const cap = RESOURCE_CAPS[key] as number;
    (result as any)[key] = Math.min(cap, Math.max(0, (result[key] as number)));
  }
  return result;
}
