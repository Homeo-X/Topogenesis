/** Initial settlement resource states for different region types. */

import type { SettlementResources } from '../types.js';

export function createInitialResources(regionId: string): SettlementResources {
  const base: SettlementResources = {
    food: 50, wood: 30, ore: 15, cloth: 10,
    medicine: 10, tools: 20, coin: 100,
    heartwellStability: 0.8, publicMorale: 0.7, security: 0.6,
  };

  // Regional modifiers
  switch (regionId) {
    case 'crown_valley':   return { ...base, food: 70, coin: 120 };
    case 'mireglass':      return { ...base, medicine: 20, food: 40 };
    case 'ash_ring':       return { ...base, ore: 40, tools: 30, food: 30 };
    case 'saffron_coast':  return { ...base, food: 55, coin: 150, cloth: 20 };
    case 'highroot':       return { ...base, wood: 50, food: 45, medicine: 15 };
    case 'quiet_north':    return { ...base, food: 30, ore: 25, heartwellStability: 0.6 };
    default:               return base;
  }
}
