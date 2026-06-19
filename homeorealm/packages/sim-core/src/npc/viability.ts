/** Viability computation and action scoring for Homeo-X loop. */

import type { NPCState, NPCNeeds, ActionType, Intention, HouseholdState, SettlementResources } from '../types.js';
import { computeViability, NEED_WEIGHTS } from './needs.js';
import { affectActionModifier } from './affect.js';
import type { SeededRng } from '../rng.js';

export type ActionCandidate = {
  action: ActionType;
  targetId?: string;
  targetResource?: string;
  motivation: string;
  baseUtility: number;
};

/** Score an action candidate against current NPC state. Higher = more likely to be chosen. */
export function scoreAction(
  npc: NPCState,
  candidate: ActionCandidate,
  household: HouseholdState | undefined,
  settlementResources: SettlementResources,
): number {
  let score = candidate.baseUtility;

  // Need pressure bonuses — actions that address deficits score higher
  const needs = npc.needs;
  switch (candidate.action) {
    case 'eat':
    case 'seek_food':
    case 'cook':
      score += (1 - needs.hunger) * NEED_WEIGHTS.hunger * 2;
      break;
    case 'rest':
    case 'sleep':
      score += (1 - needs.rest) * NEED_WEIGHTS.rest;
      break;
    case 'respond_to_danger':
    case 'seek_healer':
      score += (1 - needs.safety) * NEED_WEIGHTS.safety * 2;
      break;
    case 'social_visit':
    case 'attend_festival':
      score += (1 - needs.belonging) * NEED_WEIGHTS.belonging;
      break;
    case 'work_job':
    case 'train_skill':
    case 'craft_item':
      score += (1 - needs.purpose) * NEED_WEIGHTS.purpose;
      break;
    case 'help_household':
      score += (1 - needs.belonging) * 0.5 + (1 - needs.purpose) * 0.5;
      break;
    case 'gather_resource':
      score += (1 - needs.purpose) * 0.5;
      if (settlementResources.food < 3 && candidate.targetResource === 'food') score += 1.5;
      break;
  }

  // Frame distinction modifiers
  for (const distinction of npc.frame.distinctions) {
    if (candidate.targetId && distinction.label.includes(candidate.targetId)) {
      score *= 1 + distinction.utilityWeight * distinction.confidence;
    }
  }

  // Attention bias from frame
  for (const [tag, bias] of Object.entries(npc.frame.attentionBias)) {
    if (candidate.motivation.includes(tag)) score *= 1 + bias;
  }

  // Affect modifier
  score *= affectActionModifier(npc.affect, candidate.action);

  // Taboo avoidance
  if (npc.frame.tabooTags.some(t => candidate.motivation.includes(t))) score *= 0.1;

  // Health penalty
  if (npc.health < 0.4 && !['seek_healer', 'rest', 'sleep'].includes(candidate.action)) {
    score *= npc.health * 1.5;
  }

  return Math.max(0, score);
}

/** Generate candidate actions for a given NPC state. */
export function generateCandidates(
  npc: NPCState,
  household: HouseholdState | undefined,
  settlementResources: SettlementResources,
  day: number,
): ActionCandidate[] {
  const candidates: ActionCandidate[] = [];

  // Always available
  candidates.push({ action: 'idle', baseUtility: 0.1, motivation: 'rest from choices' });
  candidates.push({ action: 'rest', baseUtility: 0.3, motivation: 'restore stamina' });

  // Food-related
  candidates.push({ action: 'eat', baseUtility: 0.5, motivation: 'satisfy hunger', targetResource: 'food' });
  if (npc.needs.hunger < 0.3) {
    candidates.push({ action: 'seek_food', baseUtility: 0.7, motivation: 'desperate hunger', targetResource: 'food' });
  }
  if (npc.skills['cooking'] ?? 0 > 0.2) {
    candidates.push({ action: 'cook', baseUtility: 0.4, motivation: 'prepare meal', targetResource: 'food' });
  }

  // Work — if NPC has a job
  if (npc.jobId) {
    candidates.push({ action: 'work_job', baseUtility: 0.6, motivation: `fulfill role as ${npc.jobId}` });
  }

  // Social
  const friends = Object.values(npc.relationships).filter(r => r.affection > 0.2 && r.familiarity > 0.3);
  if (friends.length > 0) {
    const friend = friends[0]!;
    candidates.push({ action: 'social_visit', baseUtility: 0.3, motivation: 'maintain bonds', targetId: friend.targetId });
  }

  // Household support
  if (npc.householdId && household) {
    if (household.foodStores < 3) {
      candidates.push({ action: 'help_household', baseUtility: 0.5, motivation: 'household food shortage', targetResource: 'food' });
    }
    candidates.push({ action: 'help_household', baseUtility: 0.25, motivation: 'household contribution' });
  }

  // Skill training
  const weakSkill = Object.entries(npc.skills).find(([, v]) => v < 0.5);
  if (weakSkill) {
    candidates.push({ action: 'train_skill', baseUtility: 0.3, motivation: `improve ${weakSkill[0]}` });
  }

  // Crafting
  if ((npc.skills['crafting'] ?? 0) > 0.3) {
    candidates.push({ action: 'craft_item', baseUtility: 0.35, motivation: 'produce goods' });
  }

  // Resource gathering
  candidates.push({ action: 'gather_resource', baseUtility: 0.3, motivation: 'gather food', targetResource: 'food' });

  // Safety
  if (npc.needs.safety < 0.4) {
    candidates.push({ action: 'respond_to_danger', baseUtility: 0.8, motivation: 'seek safety' });
  }
  if (npc.health < 0.5) {
    candidates.push({ action: 'seek_healer', baseUtility: 0.7, motivation: 'restore health' });
  }

  // Settlement conditions trigger migration
  const viability = computeViability(npc);
  if (viability < 0.25 && day > 10) {
    candidates.push({ action: 'migrate', baseUtility: 0.4, motivation: 'seek better conditions' });
  }

  // Generate request for help (creates emergent quests)
  if (viability < 0.35) {
    candidates.push({ action: 'generate_request', baseUtility: 0.5, motivation: 'seek community help' });
  }

  return candidates;
}

/** Select the best intention using deterministic weighted random. */
export function selectIntention(
  npc: NPCState,
  household: HouseholdState | undefined,
  settlementResources: SettlementResources,
  day: number,
  rng: SeededRng,
): Intention {
  const candidates = generateCandidates(npc, household, settlementResources, day);
  const scores = candidates.map(c => scoreAction(npc, c, household, settlementResources));
  const total = scores.reduce((a, b) => a + b, 0);

  if (total === 0) {
    return { action: 'idle', motivation: 'no viable options', expectedViabilityGain: 0 };
  }

  const chosen = rng.nextWeighted(candidates, scores);
  const chosenScore = scores[candidates.indexOf(chosen)] ?? 0;
  const currentViability = computeViability(npc);

  return {
    action: chosen.action,
    targetId: chosen.targetId,
    targetResource: chosen.targetResource,
    motivation: chosen.motivation,
    expectedViabilityGain: (chosenScore / total) * (1 - currentViability) * 0.5,
  };
}
