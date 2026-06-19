/** NPC factory and state management utilities. */

import { nextId } from '../ids.js';
import { createRelationship } from './relationships.js';
import type { NPCState, NPCNeeds, NPCAffect, NPCFrameState } from '../types.js';

export type NPCArchetype = {
  id: string;
  people: string;
  possibleJobs: string[];
  traitBias: Record<string, number>;
  skillBias: Record<string, number>;
  namePool: string[];
};

const DEFAULT_NEEDS: NPCNeeds = {
  hunger: 0.7, rest: 0.7, safety: 0.7,
  belonging: 0.6, purpose: 0.6, autonomy: 0.6, esteem: 0.5,
};

const DEFAULT_AFFECT: NPCAffect = {
  valence: 0.2, arousal: 0.2, stress: 0.2, trustBaseline: 0.4,
};

const DEFAULT_FRAME: NPCFrameState = {
  distinctions: [],
  attentionBias: {},
  tabooTags: [],
  valuedTags: [],
};

export function createNPC(
  archetype: NPCArchetype,
  settlementId: string,
  householdId: string | undefined,
  jobId: string,
  birthDay: number,
  nameOverride?: string,
): NPCState {
  const name = nameOverride ?? archetype.namePool[Math.floor(Math.random() * archetype.namePool.length)] ?? 'Unnamed';
  const id = nextId('npc');

  return {
    id,
    name,
    ageDays: 365 * 20 + Math.floor(Math.random() * 365 * 20), // 20–40 years old
    people: archetype.people,
    householdId,
    settlementId,
    roleIds: [jobId],
    jobId,
    skills: {
      farming: archetype.skillBias['farming'] ?? 0.1,
      gathering: archetype.skillBias['gathering'] ?? 0.1,
      crafting: archetype.skillBias['crafting'] ?? 0.1,
      combat: archetype.skillBias['combat'] ?? 0.1,
      medicine: archetype.skillBias['medicine'] ?? 0.1,
      trade: archetype.skillBias['trade'] ?? 0.1,
      cooking: archetype.skillBias['cooking'] ?? 0.2,
      scholarship: archetype.skillBias['scholarship'] ?? 0.05,
      governance: archetype.skillBias['governance'] ?? 0.05,
      exploration: archetype.skillBias['exploration'] ?? 0.05,
      labor: archetype.skillBias['labor'] ?? 0.2,
    },
    traits: { ...archetype.traitBias },
    needs: { ...DEFAULT_NEEDS },
    affect: { ...DEFAULT_AFFECT },
    relationships: {},
    memories: [],
    obligations: [],
    inventory: [],
    wealth: 5 + Math.floor(Math.random() * 20),
    health: 0.8 + Math.random() * 0.2,
    reputation: { [settlementId]: 0.5 },
    frame: structuredClone(DEFAULT_FRAME),
    isAlive: true,
    birthDay,
  };
}

export function addRelationship(npc: NPCState, targetId: string, kinshipType: import('../types.js').KinshipType = 'acquaintance'): NPCState {
  return {
    ...npc,
    relationships: { ...npc.relationships, [targetId]: createRelationship(targetId, kinshipType) },
  };
}

export function applyNeedUpdates(npc: NPCState, updates: Partial<NPCNeeds>): NPCState {
  return { ...npc, needs: { ...npc.needs, ...updates } };
}
