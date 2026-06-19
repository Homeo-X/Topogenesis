/** Household creation and resource management. */

import { nextId } from '../ids.js';
import type { HouseholdState, NPCState } from '../types.js';

export function createHousehold(
  settlementId: string,
  memberIds: string[],
  name: string,
): HouseholdState {
  return {
    id: nextId('hh'),
    name,
    memberIds,
    settlementId,
    sharedInventory: [],
    foodStores: 5,  // start with moderate food
    wealth: 10 + Math.random() * 20,
    homeQuality: 0.5,
    reputation: 0.5,
    inheritanceRules: 'eldest_heir',
    lineage: [],
  };
}

/** Household daily food consumption. Returns food remaining. */
export function consumeHouseholdFood(household: HouseholdState, memberCount: number): HouseholdState {
  const consumption = memberCount * 0.15;
  const newFood = Math.max(0, household.foodStores - consumption);
  return { ...household, foodStores: newFood };
}

/** Apply delta to household food stores. */
export function adjustHouseholdFood(household: HouseholdState, delta: number): HouseholdState {
  return { ...household, foodStores: Math.max(0, Math.min(10, household.foodStores + delta)) };
}

/** Get households with food shortages. */
export function getHouseholdsWithShortage(households: Record<string, HouseholdState>): HouseholdState[] {
  return Object.values(households).filter(h => h.foodStores < 2);
}
