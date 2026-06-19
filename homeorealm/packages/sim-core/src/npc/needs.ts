/** NPC needs decay and restoration logic. */

import type { NPCNeeds, NPCState, HouseholdState, SettlementResources } from '../types.js';
import type { SeasonalModifiers } from '../time.js';

export const NEED_DECAY_RATES: Record<keyof NPCNeeds, number> = {
  hunger: 0.12,    // decays quickly without food
  rest: 0.10,      // decays with activity
  safety: 0.04,    // decays from environmental pressure
  belonging: 0.05, // decays with isolation
  purpose: 0.04,   // decays without meaningful work
  autonomy: 0.03,  // decays from over-obligation
  esteem: 0.03,    // decays from reputation loss
};

export const NEED_WEIGHTS: Record<keyof NPCNeeds, number> = {
  hunger: 2.0,    // highest weight — existential
  rest: 1.5,
  safety: 1.8,
  belonging: 1.0,
  purpose: 0.8,
  autonomy: 0.6,
  esteem: 0.6,
};

/** Apply per-tick decay to needs. Returns new needs. */
export function decayNeeds(
  needs: NPCNeeds,
  seasonal: SeasonalModifiers,
  isWorking: boolean,
): NPCNeeds {
  const illnessMod = seasonal.illnessRisk;
  return {
    hunger: Math.max(0, needs.hunger - NEED_DECAY_RATES.hunger * (isWorking ? 1.2 : 1.0)),
    rest: Math.max(0, needs.rest - NEED_DECAY_RATES.rest * (isWorking ? 1.3 : 0.7)),
    safety: Math.max(0, needs.safety - NEED_DECAY_RATES.safety * illnessMod),
    belonging: Math.max(0, needs.belonging - NEED_DECAY_RATES.belonging),
    purpose: Math.max(0, needs.purpose - NEED_DECAY_RATES.purpose),
    autonomy: Math.max(0, needs.autonomy - NEED_DECAY_RATES.autonomy),
    esteem: Math.max(0, needs.esteem - NEED_DECAY_RATES.esteem),
  };
}

/** Compute the NPC's viability score: weighted average of needs × health. */
export function computeViability(npc: NPCState): number {
  const needs = npc.needs;
  const totalWeight = Object.values(NEED_WEIGHTS).reduce((a, b) => a + b, 0);
  let weightedSum = 0;
  for (const key of Object.keys(NEED_WEIGHTS) as Array<keyof NPCNeeds>) {
    weightedSum += needs[key] * NEED_WEIGHTS[key];
  }
  const needsScore = weightedSum / totalWeight;
  return Math.max(0, Math.min(1, needsScore * npc.health));
}

/** Restore hunger from household food stores. Returns updated needs and consumed amount. */
export function consumeFood(
  needs: NPCNeeds,
  household: HouseholdState | undefined,
  settlementResources: SettlementResources,
): { needs: NPCNeeds; consumed: number } {
  if (needs.hunger >= 0.8) return { needs, consumed: 0 };

  const hungerGain = 0.4;
  let consumed = 0;

  if (household && household.foodStores > 0.5) {
    consumed = Math.min(1, hungerGain);
    return { needs: { ...needs, hunger: Math.min(1, needs.hunger + hungerGain) }, consumed };
  } else if (settlementResources.food > 2) {
    consumed = Math.min(0.5, hungerGain * 0.5);
    return { needs: { ...needs, hunger: Math.min(1, needs.hunger + hungerGain * 0.5) }, consumed };
  }
  return { needs, consumed: 0 };
}

/** Restoration amounts for non-food needs from actions. */
export const ACTION_NEED_GAINS: Partial<Record<string, Partial<NPCNeeds>>> = {
  rest:          { rest: 0.4 },
  sleep:         { rest: 0.7 },
  social_visit:  { belonging: 0.3, esteem: 0.1 },
  work_job:      { purpose: 0.3, esteem: 0.1 },
  train_skill:   { purpose: 0.2, autonomy: 0.1 },
  attend_festival: { belonging: 0.4, esteem: 0.2, purpose: 0.1 },
  help_household: { belonging: 0.2, purpose: 0.2, esteem: 0.1 },
};
