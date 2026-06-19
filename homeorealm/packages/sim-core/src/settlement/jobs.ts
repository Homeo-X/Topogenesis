/** Job definitions and assignment logic. */

export type JobDefinition = {
  id: string;
  name: string;
  description: string;
  primarySkill: string;
  produces: string[];
  requiredResources: string[];
  settlementNeed: 'essential' | 'important' | 'optional';
};

export const JOBS: Record<string, JobDefinition> = {
  farmer: {
    id: 'farmer', name: 'Farmer', description: 'Tends crops and livestock; primary food producer.',
    primarySkill: 'farming', produces: ['food'], requiredResources: ['tools'],
    settlementNeed: 'essential',
  },
  gatherer: {
    id: 'gatherer', name: 'Gatherer', description: 'Collects wild food, herbs, and wood.',
    primarySkill: 'gathering', produces: ['food', 'wood', 'medicine'],
    requiredResources: [], settlementNeed: 'essential',
  },
  guard: {
    id: 'guard', name: 'Guard', description: 'Protects settlement, patrols roads, defends against threats.',
    primarySkill: 'combat', produces: ['security'], requiredResources: ['tools'],
    settlementNeed: 'essential',
  },
  crafter: {
    id: 'crafter', name: 'Crafter', description: 'Produces tools, cloth, and goods from raw materials.',
    primarySkill: 'crafting', produces: ['tools', 'cloth'],
    requiredResources: ['wood', 'ore'], settlementNeed: 'important',
  },
  healer: {
    id: 'healer', name: 'Healer', description: 'Tends the sick; produces and dispenses medicine.',
    primarySkill: 'medicine', produces: ['medicine', 'health'],
    requiredResources: ['medicine'], settlementNeed: 'important',
  },
  merchant: {
    id: 'merchant', name: 'Merchant', description: 'Trades goods between settlements; brings coin and rare items.',
    primarySkill: 'trade', produces: ['coin'], requiredResources: [],
    settlementNeed: 'important',
  },
  scholar: {
    id: 'scholar', name: 'Scholar', description: 'Studies Hearthwell anomalies and teaches skills.',
    primarySkill: 'scholarship', produces: ['knowledge'],
    requiredResources: [], settlementNeed: 'optional',
  },
  civic_officer: {
    id: 'civic_officer', name: 'Civic Officer', description: 'Maintains public morale and governance.',
    primarySkill: 'governance', produces: ['publicMorale'],
    requiredResources: [], settlementNeed: 'important',
  },
  adventurer: {
    id: 'adventurer', name: 'Adventurer', description: 'Explores, maps, clears threats, brings rare resources.',
    primarySkill: 'exploration', produces: ['security', 'ore', 'medicine'],
    requiredResources: [], settlementNeed: 'optional',
  },
};

export function getJobsForSettlement(populationCount: number): string[] {
  const essential = Object.values(JOBS).filter(j => j.settlementNeed === 'essential').map(j => j.id);
  const important = Object.values(JOBS).filter(j => j.settlementNeed === 'important').map(j => j.id);
  const optional = Object.values(JOBS).filter(j => j.settlementNeed === 'optional').map(j => j.id);

  if (populationCount < 5) return essential;
  if (populationCount < 15) return [...essential, ...important];
  return [...essential, ...important, ...optional];
}
