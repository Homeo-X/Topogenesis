/** Execute NPC actions and emit events. */

import type { NPCState, HouseholdState, SettlementResources, Intention, WorldState } from '../types.js';
import type { EventStore, WorldEvent } from '../events.js';
import type { WorldClock } from '../time.js';
import { createMemory } from './memory.js';
import { applyInteraction } from './relationships.js';
import { ACTION_NEED_GAINS } from './needs.js';
import { nextId } from '../ids.js';

export type ActionResult = {
  needsUpdates: Partial<NPCState['needs']>;
  healthDelta: number;
  wealthDelta: number;
  skillGains: Record<string, number>;
  eventEmitted?: Omit<WorldEvent, 'id'>;
  memoryCreated?: ReturnType<typeof createMemory>;
  householdFoodDelta?: number;
  settlementResourceDeltas?: Partial<SettlementResources>;
  success: boolean;
};

/** Execute the chosen intention and compute outcomes. */
export function executeAction(
  npc: NPCState,
  intention: Intention,
  household: HouseholdState | undefined,
  settlementResources: SettlementResources,
  clock: WorldClock,
  eventStore: EventStore,
  world: WorldState,
): ActionResult {
  const baseResult: ActionResult = {
    needsUpdates: {},
    healthDelta: 0,
    wealthDelta: 0,
    skillGains: {},
    success: true,
  };

  const needGains = ACTION_NEED_GAINS[intention.action];
  if (needGains) {
    Object.assign(baseResult.needsUpdates, needGains);
  }

  switch (intention.action) {
    case 'eat':
    case 'seek_food': {
      const hasFoodInHousehold = household && household.foodStores > 0.5;
      const hasFoodInSettlement = settlementResources.food > 1;
      if (hasFoodInHousehold || hasFoodInSettlement) {
        baseResult.needsUpdates.hunger = Math.min(1, (npc.needs.hunger) + 0.4);
        baseResult.healthDelta = 0.02;
        if (hasFoodInHousehold) {
          baseResult.householdFoodDelta = -0.5;
        } else {
          baseResult.settlementResourceDeltas = { food: -0.5 };
        }
        baseResult.eventEmitted = {
          day: clock.day, tick: clock.tick, type: 'npc_ate',
          actorId: npc.id, settlementId: npc.settlementId,
          payload: { source: hasFoodInHousehold ? 'household' : 'settlement' },
          tags: ['food', 'consumption'], salience: 0.2,
        };
      } else {
        baseResult.success = false;
        baseResult.needsUpdates.hunger = Math.max(0, npc.needs.hunger - 0.05);
        baseResult.healthDelta = -0.03;
        baseResult.eventEmitted = {
          day: clock.day, tick: clock.tick, type: 'npc_hunger_crisis',
          actorId: npc.id, settlementId: npc.settlementId,
          payload: { need: 'hunger' }, tags: ['food', 'crisis', 'shortage'], salience: 0.8,
        };
        baseResult.memoryCreated = createMemory(
          npc.id, clock.day, 'hunger_crisis', 0.8, -0.7, [npc.id],
          ['hunger', 'shortage'], `Suffered from lack of food in ${npc.settlementId}`,
        );
      }
      break;
    }

    case 'work_job': {
      const job = npc.jobId ?? 'laborer';
      const skillKey = JOB_SKILL_MAP[job] ?? 'labor';
      const skill = npc.skills[skillKey] ?? 0.1;
      const output = skill * 0.5 + 0.2;
      baseResult.skillGains[skillKey] = 0.01;
      baseResult.wealthDelta = output * 0.3;
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.3);
      baseResult.settlementResourceDeltas = getJobOutput(job, output);
      baseResult.eventEmitted = {
        day: clock.day, tick: clock.tick, type: 'npc_worked',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: { job, output }, tags: ['work', job], salience: 0.2,
      };
      break;
    }

    case 'social_visit': {
      const targetId = intention.targetId;
      if (targetId && world.npcs[targetId]) {
        baseResult.needsUpdates.belonging = Math.min(1, npc.needs.belonging + 0.3);
        baseResult.needsUpdates.esteem = Math.min(1, npc.needs.esteem + 0.1);
        baseResult.eventEmitted = {
          day: clock.day, tick: clock.tick, type: 'npc_social_visit',
          actorId: npc.id, targetIds: [targetId], settlementId: npc.settlementId,
          payload: {}, tags: ['social', 'relationship'], salience: 0.3,
        };
        baseResult.memoryCreated = createMemory(
          npc.id, clock.day, 'social_visit', 0.35, 0.4, [npc.id, targetId],
          ['social', 'belonging'], `Visited ${world.npcs[targetId]?.name ?? targetId}`,
        );
      }
      break;
    }

    case 'rest':
    case 'sleep': {
      const gain = intention.action === 'sleep' ? 0.7 : 0.4;
      baseResult.needsUpdates.rest = Math.min(1, npc.needs.rest + gain);
      baseResult.healthDelta = 0.01;
      break;
    }

    case 'gather_resource': {
      const amount = 0.5 + (npc.skills['gathering'] ?? 0.1) * 0.5;
      baseResult.skillGains['gathering'] = 0.01;
      baseResult.settlementResourceDeltas = { food: amount * 0.5, wood: amount * 0.3 };
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.2);
      break;
    }

    case 'train_skill': {
      const allSkills = Object.keys(npc.skills);
      const weakest = allSkills.sort((a, b) => (npc.skills[a] ?? 0) - (npc.skills[b] ?? 0))[0] ?? 'farming';
      baseResult.skillGains[weakest] = 0.03;
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.2);
      baseResult.needsUpdates.autonomy = Math.min(1, npc.needs.autonomy + 0.1);
      break;
    }

    case 'craft_item': {
      baseResult.skillGains['crafting'] = 0.02;
      baseResult.wealthDelta = 0.4;
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.25);
      baseResult.settlementResourceDeltas = { tools: 0.3 };
      break;
    }

    case 'seek_healer': {
      const healerExists = Object.values(world.npcs).some(n => n.settlementId === npc.settlementId && n.jobId === 'healer' && n.isAlive);
      if (healerExists && settlementResources.medicine > 0.5) {
        baseResult.healthDelta = 0.15;
        baseResult.needsUpdates.safety = Math.min(1, npc.needs.safety + 0.2);
        baseResult.settlementResourceDeltas = { medicine: -0.3 };
        baseResult.eventEmitted = {
          day: clock.day, tick: clock.tick, type: 'npc_healed',
          actorId: npc.id, settlementId: npc.settlementId,
          payload: { healthGain: 0.15 }, tags: ['health', 'healing'], salience: 0.5,
        };
        baseResult.memoryCreated = createMemory(
          npc.id, clock.day, 'healing', 0.6, 0.5, [npc.id],
          ['health', 'healing', 'recovery'], `Received healing care`,
        );
      }
      break;
    }

    case 'help_household': {
      if (household) {
        baseResult.needsUpdates.belonging = Math.min(1, npc.needs.belonging + 0.2);
        baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.2);
        baseResult.householdFoodDelta = 0.5;
        baseResult.eventEmitted = {
          day: clock.day, tick: clock.tick, type: 'npc_helped_household',
          actorId: npc.id, settlementId: npc.settlementId,
          payload: { householdId: household.id }, tags: ['household', 'social', 'help'], salience: 0.3,
        };
      }
      break;
    }

    case 'attend_festival': {
      baseResult.needsUpdates.belonging = Math.min(1, npc.needs.belonging + 0.4);
      baseResult.needsUpdates.esteem = Math.min(1, npc.needs.esteem + 0.2);
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.1);
      baseResult.eventEmitted = {
        day: clock.day, tick: clock.tick, type: 'npc_attended_festival',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: {}, tags: ['festival', 'social', 'ritual'], salience: 0.6,
      };
      baseResult.memoryCreated = createMemory(
        npc.id, clock.day, 'festival', 0.65, 0.6, [npc.id],
        ['festival', 'belonging', 'community'], `Attended seasonal festival`,
      );
      break;
    }

    case 'generate_request': {
      baseResult.needsUpdates.autonomy = Math.min(1, npc.needs.autonomy + 0.1);
      baseResult.eventEmitted = {
        day: clock.day, tick: clock.tick, type: 'npc_requested_help',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: { needs: npc.needs }, tags: ['quest_trigger', 'help', 'viability'], salience: 0.75,
      };
      break;
    }

    case 'respond_to_danger': {
      baseResult.needsUpdates.safety = Math.min(1, npc.needs.safety + 0.3);
      break;
    }

    case 'migrate': {
      baseResult.eventEmitted = {
        day: clock.day, tick: clock.tick, type: 'npc_migration',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: { reason: 'low_viability' }, tags: ['migration', 'viability_crisis'], salience: 0.85,
      };
      baseResult.memoryCreated = createMemory(
        npc.id, clock.day, 'migration', 0.9, -0.3, [npc.id],
        ['migration', 'hardship', 'change'], `Left settlement due to poor conditions`,
      );
      break;
    }

    case 'cook': {
      const cookSkill = npc.skills['cooking'] ?? 0.1;
      const foodProduced = 0.8 + cookSkill * 0.5;
      baseResult.skillGains['cooking'] = 0.01;
      baseResult.householdFoodDelta = foodProduced;
      baseResult.needsUpdates.purpose = Math.min(1, npc.needs.purpose + 0.2);
      break;
    }
  }

  // Emit event if present
  if (baseResult.eventEmitted) {
    eventStore.append(baseResult.eventEmitted);
  }

  return baseResult;
}

const JOB_SKILL_MAP: Record<string, string> = {
  farmer: 'farming', gatherer: 'gathering', guard: 'combat',
  crafter: 'crafting', healer: 'medicine', merchant: 'trade',
  scholar: 'scholarship', civic_officer: 'governance', adventurer: 'exploration',
};

function getJobOutput(job: string, amount: number): Partial<SettlementResources> {
  switch (job) {
    case 'farmer':   return { food: amount * 1.5 };
    case 'gatherer': return { food: amount * 0.8, wood: amount * 0.5 };
    case 'guard':    return { security: amount * 0.1 };
    case 'crafter':  return { tools: amount * 0.8, cloth: amount * 0.3 };
    case 'healer':   return { medicine: amount * 0.7 };
    case 'merchant': return { coin: amount * 0.6 };
    case 'scholar':  return {};
    case 'civic_officer': return { publicMorale: amount * 0.05 };
    default:         return { food: amount * 0.3 };
  }
}
