import type { EventStore } from '../events.js';
import type { EmergentQuest, PlayerInventoryItem, PlayerState, QuestObjective, WorldState } from '../types.js';
import { completeQuest } from '../quests/questTypes.js';

export type PlayerActionType =
  | 'accept_quest'
  | 'complete_quest'
  | 'aid_settlement'
  | 'travel'
  | 'gather'
  | 'rest'
  | 'train'
  | 'trade'
  | 'talk'
  | 'delve_dungeon';

export type PlayerActionInput = {
  type: PlayerActionType;
  questId?: string;
  settlementId?: string;
  location?: PlayerState['location'];
  npcId?: string;
};

export type PlayerActionResult = {
  world: WorldState;
  player: PlayerState;
  message: string;
};

export function createPlayer(settlementId: string, name = 'Wayfarer'): PlayerState {
  return {
    id: 'player_1',
    name,
    settlementId,
    location: 'town',
    health: 1,
    stamina: 1,
    level: 1,
    experience: 0,
    skills: { gathering: 0.1, combat: 0.08, craft: 0.08, social: 0.12, trade: 0.1 },
    wealth: 20,
    reputation: { [settlementId]: 0.1 },
    inventory: [],
    questLog: [],
    actionLog: ['Arrived at the settlement.'],
  };
}

export function ensurePlayer(world: WorldState): PlayerState {
  const settlementId = Object.keys(world.settlements)[0] ?? 'unknown_settlement';
  if (!world.player) return createPlayer(settlementId);
  const fallback = createPlayer(world.player.settlementId ?? settlementId);
  return {
    ...fallback,
    ...world.player,
    location: world.player.location ?? fallback.location,
    health: world.player.health ?? fallback.health,
    stamina: world.player.stamina ?? fallback.stamina,
    level: world.player.level ?? fallback.level,
    experience: world.player.experience ?? fallback.experience,
    skills: { ...fallback.skills, ...(world.player.skills ?? {}) },
  };
}

export function applyPlayerAction(
  world: WorldState,
  eventStore: EventStore,
  input: PlayerActionInput,
): PlayerActionResult {
  const player = ensurePlayer(world);

  if (input.type === 'accept_quest') {
    if (!input.questId) throw new Error('questId is required');
    const quest = world.quests[input.questId];
    if (!quest || !quest.isActive) throw new Error('Quest is not active');
    if (player.questLog.some(q => q.questId === quest.id)) throw new Error('Quest already accepted');

    const updatedPlayer: PlayerState = {
      ...player,
      settlementId: quest.settlementId,
      questLog: [...player.questLog, { questId: quest.id, status: 'accepted', acceptedOnDay: world.day }],
      actionLog: [`Accepted quest: ${quest.title}`, ...player.actionLog].slice(0, 20),
    };
    const updatedWorld: WorldState = {
      ...world,
      player: updatedPlayer,
      quests: { ...world.quests, [quest.id]: { ...quest, acceptedByPlayerId: player.id } },
    };

    eventStore.append({
      day: world.day,
      tick: world.tick,
      type: 'player_quest_accepted',
      actorId: player.id,
      settlementId: quest.settlementId,
      payload: { questId: quest.id, title: quest.title },
      tags: ['player', 'quest', 'accepted'],
      salience: 0.7,
    });

    return { world: updatedWorld, player: updatedPlayer, message: `Accepted "${quest.title}".` };
  }

  if (input.type === 'complete_quest') {
    if (!input.questId) throw new Error('questId is required');
    const quest = world.quests[input.questId];
    if (!quest || !quest.isActive) throw new Error('Quest is not active');
    const questRecord = player.questLog.find(q => q.questId === quest.id && q.status === 'accepted');
    if (!questRecord) throw new Error('Quest must be accepted before completion');
    if (!quest.objectives.every(o => o.completed)) throw new Error('Quest objectives are not complete yet');

    const settlement = world.settlements[quest.settlementId];
    const coinReward = quest.rewards.filter(r => r.type === 'coin').reduce((sum, r) => sum + r.value, 0);
    const reputationReward = quest.rewards.filter(r => r.type === 'reputation').reduce((sum, r) => sum + r.value, 0);
    const currentRep = player.reputation[quest.settlementId] ?? 0;
    const nextRep = Math.max(0, Math.min(1, currentRep + Math.max(0.03, reputationReward / 100)));
    const moraleGain = Math.min(0.08, 0.02 + quest.urgency * 0.04);

    const updatedPlayer: PlayerState = {
      ...player,
      wealth: player.wealth + coinReward,
      reputation: { ...player.reputation, [quest.settlementId]: nextRep },
      questLog: player.questLog.map(q => q.questId === quest.id ? { ...q, status: 'completed', completedOnDay: world.day } : q),
      actionLog: [`Completed quest: ${quest.title}`, ...player.actionLog].slice(0, 20),
    };

    const updatedWorld: WorldState = {
      ...world,
      player: updatedPlayer,
      quests: {
        ...world.quests,
        [quest.id]: { ...completeQuest(quest), completedByPlayerId: player.id },
      },
      settlements: settlement
        ? {
            ...world.settlements,
            [settlement.id]: {
              ...settlement,
              resources: {
                ...settlement.resources,
                publicMorale: Math.min(1, settlement.resources.publicMorale + moraleGain),
              },
            },
          }
        : world.settlements,
    };

    eventStore.append({
      day: world.day,
      tick: world.tick,
      type: 'player_quest_completed',
      actorId: player.id,
      settlementId: quest.settlementId,
      payload: { questId: quest.id, title: quest.title, coinReward, reputation: nextRep },
      tags: ['player', 'quest', 'completed'],
      salience: 0.9,
    });

    return { world: updatedWorld, player: updatedPlayer, message: `Completed "${quest.title}".` };
  }

  const settlementId = input.settlementId ?? player.settlementId;
  const settlement = world.settlements[settlementId];
  if (!settlement) throw new Error('Settlement not found');

  if (input.type === 'travel') {
    const location = input.location ?? 'town';
    const updatedPlayer = withLog({ ...player, settlementId, location, stamina: Math.max(0, player.stamina - 0.04) }, `Traveled to ${label(location)}.`);
    const updatedWorld = { ...world, player: updatedPlayer };
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_traveled', { location }, ['player', 'travel'], 0.45);
    return { world: updatedWorld, player: updatedPlayer, message: `Traveled to ${label(location)}.` };
  }

  if (input.type === 'rest') {
    const updatedPlayer = withLog({
      ...player,
      location: 'home',
      stamina: Math.min(1, player.stamina + 0.45),
      health: Math.min(1, player.health + 0.12),
    }, 'Rested at home.');
    const updatedWorld = { ...world, player: updatedPlayer };
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_rested', { stamina: updatedPlayer.stamina, health: updatedPlayer.health }, ['player', 'rest'], 0.45);
    return { world: updatedWorld, player: updatedPlayer, message: 'Rested and recovered.' };
  }

  if (input.type === 'gather') {
    requireStamina(player, 0.15);
    const updatedPlayer = gainExperience(withInventory(withSkill({
      ...player,
      location: 'wilds',
      stamina: Math.max(0, player.stamina - 0.15),
    }, 'gathering', 0.03), 'foraged_food', 3), 8);
    const updatedWorld = progressAcceptedQuests({
      ...world,
      player: withLog(updatedPlayer, 'Gathered food in the wilds.'),
      settlements: {
        ...world.settlements,
        [settlementId]: {
          ...settlement,
          resources: { ...settlement.resources, food: settlement.resources.food + 2 },
        },
      },
    }, player.id, 'gather');
    appendPlayerEvent(eventStore, world, updatedWorld.player!, 'player_gathered', { itemId: 'foraged_food', quantity: 3 }, ['player', 'gather'], 0.55);
    return { world: updatedWorld, player: updatedWorld.player!, message: 'Gathered food and stocked the settlement.' };
  }

  if (input.type === 'train') {
    requireStamina(player, 0.18);
    const updatedPlayer = withLog(gainExperience(withSkill({
      ...player,
      location: 'town',
      stamina: Math.max(0, player.stamina - 0.18),
    }, 'combat', 0.025), 10), 'Trained with the town guard.');
    const updatedWorld = progressAcceptedQuests({ ...world, player: updatedPlayer }, player.id, 'guard');
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_trained', { skill: 'combat' }, ['player', 'training'], 0.5);
    return { world: updatedWorld, player: updatedPlayer, message: 'Training improved combat skill.' };
  }

  if (input.type === 'trade') {
    const food = player.inventory.find(i => i.itemId === 'foraged_food')?.quantity ?? 0;
    if (food <= 0) throw new Error('No foraged food to trade');
    const sellQuantity = Math.min(3, food);
    const updatedPlayer = withLog(gainExperience(withSkill({
      ...player,
      location: 'market',
      wealth: player.wealth + sellQuantity * 2,
      inventory: addItem(player.inventory, 'foraged_food', -sellQuantity),
      reputation: { ...player.reputation, [settlementId]: Math.min(1, (player.reputation[settlementId] ?? 0) + 0.01) },
    }, 'trade', 0.02), 6), `Traded ${sellQuantity} food at market.`);
    const updatedWorld = progressAcceptedQuests({ ...world, player: updatedPlayer }, player.id, 'trade');
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_traded', { itemId: 'foraged_food', quantity: sellQuantity }, ['player', 'trade'], 0.5);
    return { world: updatedWorld, player: updatedPlayer, message: `Sold ${sellQuantity} food at market.` };
  }

  if (input.type === 'talk') {
    const npc = input.npcId ? world.npcs[input.npcId] : undefined;
    const updatedPlayer = withLog(gainExperience(withSkill({
      ...player,
      location: 'town',
      stamina: Math.max(0, player.stamina - 0.05),
      reputation: { ...player.reputation, [settlementId]: Math.min(1, (player.reputation[settlementId] ?? 0) + 0.012) },
    }, 'social', 0.018), 5), npc ? `Talked with ${npc.name}.` : 'Talked with townsfolk.');
    const updatedWorld = progressAcceptedQuests({ ...world, player: updatedPlayer }, player.id, 'mediate');
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_talked', { npcId: npc?.id, npcName: npc?.name }, ['player', 'social'], 0.5);
    return { world: updatedWorld, player: updatedPlayer, message: npc ? `${npc.name} warms to your presence.` : 'You learned what the town needs.' };
  }

  if (input.type === 'delve_dungeon') {
    requireStamina(player, 0.25);
    const difficulty = world.dungeonRooms[0]?.difficulty ?? 0.35;
    const combat = player.skills.combat ?? 0;
    const damage = Math.max(0.03, difficulty * 0.2 - combat * 0.08);
    const foundRelic = difficulty > 0.45 || combat > 0.2;
    const updatedPlayer = withLog(gainExperience(withInventory(withSkill({
      ...player,
      location: 'dungeon',
      stamina: Math.max(0, player.stamina - 0.25),
      health: Math.max(0.05, player.health - damage),
      wealth: player.wealth + Math.round(4 + difficulty * 10),
    }, 'combat', 0.035), foundRelic ? 'memory_relic' : 'old_coin', 1), 18), foundRelic ? 'Returned from a dungeon with a memory relic.' : 'Scouted a dungeon and found old coin.');
    const updatedWorld = progressAcceptedQuests({
      ...world,
      player: updatedPlayer,
      settlements: {
        ...world.settlements,
        [settlementId]: {
          ...settlement,
          resources: {
            ...settlement.resources,
            security: Math.min(1, settlement.resources.security + 0.02),
            publicMorale: Math.min(1, settlement.resources.publicMorale + 0.015),
          },
        },
      },
    }, player.id, 'explore');
    appendPlayerEvent(eventStore, world, updatedPlayer, 'player_delve_completed', { damage, foundRelic }, ['player', 'dungeon'], 0.75);
    return { world: updatedWorld, player: updatedPlayer, message: foundRelic ? 'You recovered a memory relic.' : 'You returned with coin and bruises.' };
  }

  const updatedPlayer: PlayerState = {
    ...player,
    settlementId,
    reputation: {
      ...player.reputation,
      [settlementId]: Math.min(1, (player.reputation[settlementId] ?? 0) + 0.02),
    },
    actionLog: [`Aided ${settlement.name}'s daily needs.`, ...player.actionLog].slice(0, 20),
  };
  const updatedWorld: WorldState = {
    ...world,
    player: updatedPlayer,
    settlements: {
      ...world.settlements,
      [settlementId]: {
        ...settlement,
        resources: {
          ...settlement.resources,
          food: settlement.resources.food + 3,
          publicMorale: Math.min(1, settlement.resources.publicMorale + 0.015),
        },
      },
    },
  };

  eventStore.append({
    day: world.day,
    tick: world.tick,
    type: 'player_aided_settlement',
    actorId: player.id,
    settlementId,
    payload: { settlementName: settlement.name, foodAdded: 3 },
    tags: ['player', 'settlement', 'aid'],
    salience: 0.55,
  });

  return { world: updatedWorld, player: updatedPlayer, message: `Aided ${settlement.name}.` };
}

type ObjectiveProgressAction = 'gather' | 'trade' | 'guard' | 'mediate' | 'explore';

const ACTION_OBJECTIVE_TYPES: Record<ObjectiveProgressAction, QuestObjective['type'][]> = {
  gather: ['gather', 'deliver'],
  trade: ['deliver'],
  guard: ['guard'],
  mediate: ['mediate'],
  explore: ['explore', 'investigate'],
};

function progressAcceptedQuests(world: WorldState, playerId: string, action: ObjectiveProgressAction): WorldState {
  const objectiveTypes = ACTION_OBJECTIVE_TYPES[action];
  let changed = false;
  const quests: Record<string, EmergentQuest> = {};

  for (const [questId, quest] of Object.entries(world.quests)) {
    if (!quest.isActive || quest.acceptedByPlayerId !== playerId) {
      quests[questId] = quest;
      continue;
    }

    let progressed = false;
    const objectives = quest.objectives.map(objective => {
      if (objective.completed || !objectiveTypes.includes(objective.type)) return objective;
      progressed = true;
      return { ...objective, completed: true };
    });

    if (progressed) {
      changed = true;
      quests[questId] = { ...quest, objectives };
    } else {
      quests[questId] = quest;
    }
  }

  return changed ? { ...world, quests } : world;
}

function label(location: PlayerState['location']): string {
  return location.replace(/_/g, ' ');
}

function requireStamina(player: PlayerState, amount: number): void {
  if (player.stamina < amount) throw new Error('Too tired. Rest before doing that.');
}

function withLog(player: PlayerState, entry: string): PlayerState {
  return { ...player, actionLog: [entry, ...player.actionLog].slice(0, 20) };
}

function withSkill(player: PlayerState, skill: string, gain: number): PlayerState {
  return {
    ...player,
    skills: { ...player.skills, [skill]: Math.min(1, (player.skills[skill] ?? 0) + gain) },
  };
}

function gainExperience(player: PlayerState, amount: number): PlayerState {
  const experience = player.experience + amount;
  const nextLevel = Math.max(player.level, Math.floor(experience / 100) + 1);
  return { ...player, experience, level: nextLevel };
}

function withInventory(player: PlayerState, itemId: string, quantity: number): PlayerState {
  return { ...player, inventory: addItem(player.inventory, itemId, quantity) };
}

function addItem(inventory: PlayerInventoryItem[], itemId: string, quantity: number): PlayerInventoryItem[] {
  const next = [...inventory];
  const index = next.findIndex(i => i.itemId === itemId);
  if (index >= 0) {
    const updated = { ...next[index]!, quantity: next[index]!.quantity + quantity };
    if (updated.quantity <= 0) next.splice(index, 1);
    else next[index] = updated;
    return next;
  }
  if (quantity <= 0) return next;
  return [...next, { itemId, quantity }];
}

function appendPlayerEvent(
  eventStore: EventStore,
  world: WorldState,
  player: PlayerState,
  type: string,
  payload: Record<string, unknown>,
  tags: string[],
  salience: number,
): void {
  eventStore.append({
    day: world.day,
    tick: world.tick,
    type,
    actorId: player.id,
    settlementId: player.settlementId,
    payload,
    tags,
    salience,
  });
}
