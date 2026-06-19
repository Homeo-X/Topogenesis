/** Main simulation tick — advances world state by one day. */

import type { WorldState, NPCState, SettlementState } from '../types.js';
import type { EventStore } from '../events.js';
import { computeClock, getSeasonalModifiers, isFestivalDay } from '../time.js';
import { decayNeeds, consumeFood } from '../npc/needs.js';
import { updateAffect } from '../npc/affect.js';
import { decayMemories } from '../npc/memory.js';
import { driftRelationship } from '../npc/relationships.js';
import { selectIntention } from '../npc/viability.js';
import { executeAction } from '../npc/policy.js';
import { proposeDistinction, extendFrame } from '../npc/schedules.js';
import { applyDailyConsumption, applyResourceDelta } from '../settlement/economy.js';
import { adjustHouseholdFood, consumeHouseholdFood } from '../settlement/households.js';
import { generateEmergentQuests } from '../quests/questGenerator.js';
import { expireQuests } from '../quests/questTypes.js';
import { generateDungeonRooms, refreshDungeonRooms } from '../dungeon/dungeonTopogenesis.js';
import { createRng } from '../rng.js';

export type TickOptions = {
  logSummary?: boolean;
};

/** Advance the world state by exactly one day. Fully deterministic given the same seed and input state. */
export function tickDay(world: WorldState, eventStore: EventStore, opts: TickOptions = {}): WorldState {
  const day = world.day + 1;
  const seed = world.seed;
  const dayRng = createRng((seed * 1664525 + day * 22695477) >>> 0);
  const clock = computeClock(day);

  let npcs = { ...world.npcs };
  let households = { ...world.households };
  let settlements = { ...world.settlements };
  let quests = { ...world.quests };
  let dungeonRooms = [...world.dungeonRooms];

  // ─── Per-settlement daily update ───────────────────────────────────────
  for (const settlementId of Object.keys(settlements)) {
    const settlement = settlements[settlementId]!;
    const seasonal = getSeasonalModifiers(clock.season);
    const settlementRng = dayRng.fork(settlementId);

    // Resource consumption
    const { resources: updatedResources, shortages } = applyDailyConsumption(
      settlement.resources,
      settlement.population,
      seasonal,
    );

    // Emit shortage events
    for (const shortage of shortages) {
      eventStore.append({
        day, tick: 0,
        type: 'settlement_shortage',
        settlementId,
        payload: { resource: shortage },
        tags: [shortage, 'shortage', 'settlement'],
        salience: 0.7,
      });
    }

    settlements[settlementId] = { ...settlement, resources: updatedResources, day };

    // Generate quests from pressures
    const worldForQuests: WorldState = { ...world, settlements, npcs, households, quests, dungeonRooms, day };
    const newQuests = generateEmergentQuests(worldForQuests, settlementId, settlementRng);
    for (const q of newQuests) {
      quests[q.id] = q;
      eventStore.append({
        day, tick: 0, type: 'quest_generated',
        settlementId,
        payload: { questId: q.id, title: q.title, cause: q.cause },
        tags: ['quest', 'emergent', ...q.tags],
        salience: 0.5 + q.urgency * 0.3,
      });
    }

    // Refresh dungeon rooms
    dungeonRooms = refreshDungeonRooms(
      dungeonRooms.filter(r => r.settlementId !== settlementId),
      settlementId,
      updatedResources,
      settlement.population,
      day,
      settlementRng.fork('dungeon'),
    ).concat(dungeonRooms.filter(r => r.settlementId === settlementId && day - r.generatedOnDay < 10));
  }

  // ─── Expire old quests ─────────────────────────────────────────────────
  quests = expireQuests(quests, day);

  // ─── Festival detection ─────────────────────────────────────────────────
  const isFestival = isFestivalDay(clock);

  // ─── Per-NPC daily update ──────────────────────────────────────────────
  for (const npcId of Object.keys(npcs)) {
    let npc = npcs[npcId]!;
    if (!npc.isAlive) continue;

    const settlement = settlements[npc.settlementId];
    if (!settlement) continue;

    const household = npc.householdId ? households[npc.householdId] : undefined;
    const seasonal = getSeasonalModifiers(clock.season);
    const npcRng = dayRng.fork(npcId);

    // 1. Decay needs
    const isWorking = npc.currentIntention?.action === 'work_job';
    npc = { ...npc, needs: decayNeeds(npc.needs, seasonal, isWorking) };

    // 2. Consume food from household/settlement
    const { needs: needsAfterFood, consumed } = consumeFood(npc.needs, household, settlement.resources);
    npc = { ...npc, needs: needsAfterFood };
    if (consumed > 0 && household) {
      const hh = adjustHouseholdFood(household, -consumed);
      households[household.id] = hh;
    }

    // 3. Update affect based on current needs
    npc = { ...npc, affect: updateAffect(npc.affect, npc.needs) };

    // 4. Decay memories
    npc = { ...npc, memories: decayMemories(npc.memories, day) };

    // 5. Drift relationships
    const driftedRels: typeof npc.relationships = {};
    for (const [k, rel] of Object.entries(npc.relationships)) {
      driftedRels[k] = driftRelationship(rel, day);
    }
    npc = { ...npc, relationships: driftedRels };

    // 6. Festival participation
    if (isFestival && npcRng.nextBool(0.7)) {
      npc = { ...npc, currentIntention: { action: 'attend_festival', motivation: 'seasonal festival', expectedViabilityGain: 0.15 } };
    } else {
      // 7. Select intention from viability loop
      const worldSnapshot: WorldState = { ...world, settlements, npcs, households, quests, dungeonRooms, day };
      npc = { ...npc, currentIntention: selectIntention(npc, household, settlement.resources, day, npcRng) };
    }

    // 8. Execute action
    const worldForAction: WorldState = { ...world, settlements, npcs, households, quests, dungeonRooms, day };
    const result = executeAction(npc, npc.currentIntention!, household, settlement.resources, clock, eventStore, worldForAction);

    // Apply needs updates from action
    if (result.needsUpdates) {
      npc = { ...npc, needs: { ...npc.needs, ...result.needsUpdates } };
    }
    npc = { ...npc, health: Math.max(0, Math.min(1, npc.health + result.healthDelta)), wealth: npc.wealth + result.wealthDelta };

    // Apply skill gains
    if (result.skillGains) {
      const updatedSkills = { ...npc.skills };
      for (const [skill, gain] of Object.entries(result.skillGains)) {
        updatedSkills[skill] = Math.min(1, (updatedSkills[skill] ?? 0) + gain);
      }
      npc = { ...npc, skills: updatedSkills };
    }

    // Add memory if created
    if (result.memoryCreated) {
      npc = { ...npc, memories: [...npc.memories, result.memoryCreated] };
    }

    // Apply household food delta
    if (result.householdFoodDelta !== undefined && household) {
      households[household.id] = adjustHouseholdFood(households[household.id]!, result.householdFoodDelta);
    }

    // Apply settlement resource deltas from NPC actions
    if (result.settlementResourceDeltas && settlement) {
      settlements[npc.settlementId] = {
        ...settlements[npc.settlementId]!,
        resources: applyResourceDelta(settlements[npc.settlementId]!.resources, result.settlementResourceDeltas),
      };
    }

    // 9. Frame distinction learning (topogenesis)
    const recentMemories = npc.memories.filter(m => day - m.day <= 7 && m.salience > 0.5);
    const candidate = proposeDistinction(npc, recentMemories, day);
    if (candidate) {
      npc = { ...npc, frame: extendFrame(npc.frame, candidate) };
      eventStore.append({
        day, tick: 0, type: 'frame_distinction_formed',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: { distinction: candidate.label, domain: candidate.domain },
        tags: ['topogenesis', 'frame', candidate.domain],
        salience: 0.6,
      });
    }

    // 10. Health crisis → death
    if (npc.health <= 0) {
      npc = { ...npc, isAlive: false };
      eventStore.append({
        day, tick: 0, type: 'npc_death',
        actorId: npc.id, settlementId: npc.settlementId,
        payload: { cause: 'health_failure' },
        tags: ['death', 'viability'], salience: 0.95,
      });
      // Remove from settlement
      if (settlement) {
        settlements[npc.settlementId] = {
          ...settlements[npc.settlementId]!,
          npcIds: settlement.npcIds.filter(id => id !== npc.id),
          population: Math.max(0, settlement.population - 1),
        };
      }
    }

    // 11. Age
    npc = { ...npc, ageDays: npc.ageDays + 1 };

    npcs[npcId] = npc;
  }

  // ─── Household food consumption ─────────────────────────────────────────
  for (const hhId of Object.keys(households)) {
    const hh = households[hhId]!;
    const members = hh.memberIds.filter(id => npcs[id]?.isAlive).length;
    households[hhId] = consumeHouseholdFood(hh, members);
  }

  return { seed, day, tick: 0, settlements, npcs, households, quests, dungeonRooms };
}
